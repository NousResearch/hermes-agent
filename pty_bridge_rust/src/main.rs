use portable_pty::{CommandBuilder, NativePtySystem, PtySize, PtySystem};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::AsyncReadExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: hermes-pty-refactor <command> [args...]");
        std::process::exit(1);
    }

    let pty_system = NativePtySystem::default();
    let mut cmd = CommandBuilder::new(&args[0]);
    if args.len() > 1 {
        cmd.args(&args[1..]);
    }

    let pair = pty_system.openpty(PtySize {
        rows: 24,
        cols: 80,
        pixel_width: 0,
        pixel_height: 0,
    })?;

    let mut child = pair.slave.spawn_command(cmd)?;
    // Drop slave side so EOF triggers correctly
    drop(pair.slave);

    // Thread to read PTY output and write to stdout
    let is_running = Arc::new(AtomicBool::new(true));
    let is_running_clone = Arc::clone(&is_running);
    let mut pty_reader = pair.master.try_clone_reader()?;
    std::thread::spawn(move || {
        let mut buf = [0u8; 65536];
        let mut stdout = std::io::stdout();
        while let Ok(n) = pty_reader.read(&mut buf) {
            if n == 0 {
                break;
            }
            if stdout.write_all(&buf[..n]).is_err() {
                break;
            }
            let _ = stdout.flush();
        }
        is_running_clone.store(false, Ordering::SeqCst);
    });

    // Async loop to read stdin and write to PTY master input
    let resize_re = regex::bytes::Regex::new(r"^\x1b\[8;(\d+);(\d+)t$").unwrap();
    let mut stdin = tokio::io::stdin();
    let mut pty_writer = pair.master.take_writer()?;
    let mut buf = [0u8; 4096];

    loop {
        if !is_running.load(Ordering::SeqCst) {
            break;
        }
        tokio::select! {
            res = stdin.read(&mut buf) => {
                match res {
                    Ok(0) => {
                        break;
                    }
                    Ok(n) => {
                        let data = &buf[..n];
                        if let Some(caps) = resize_re.captures(data) {
                            let rows_str = std::str::from_utf8(&caps[1]).unwrap_or("24");
                            let cols_str = std::str::from_utf8(&caps[2]).unwrap_or("80");
                            let rows = rows_str.parse::<u16>().unwrap_or(24);
                            let cols = cols_str.parse::<u16>().unwrap_or(80);
                            let _ = pair.master.resize(PtySize {
                                rows,
                                cols,
                                pixel_width: 0,
                                pixel_height: 0,
                            });
                        } else {
                            if pty_writer.write_all(data).is_err() {
                                break;
                            }
                        }
                    }
                    Err(_) => {
                        break;
                    }
                }
            }
            // Monitor if child process exits or reader finished
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                if !is_running.load(Ordering::SeqCst) {
                    break;
                }
                match child.try_wait() {
                    Ok(Some(_status)) => {
                        break;
                    }
                    Err(_) => {
                        break;
                    }
                    Ok(None) => {}
                }
            }
        }
    }

    // Terminate child process if still running
    let _ = child.kill();
    std::process::exit(0);
}
