use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::process::{Command, Stdio};

/// Copy text to the system clipboard. Matches the official TUI's platform
/// fallbacks (pbcopy / wl-copy / xclip / clip) without an extra crate.
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let attempts: &[(&str, &[&str])] = if cfg!(target_os = "macos") {
        &[("pbcopy", &[])]
    } else if cfg!(target_os = "windows") {
        &[("clip", &[])]
    } else {
        &[
            ("wl-copy", &[]),
            ("xclip", &["-selection", "clipboard"]),
            ("xsel", &["--clipboard", "--input"]),
        ]
    };

    let mut last_err = None;
    for (cmd, args) in attempts {
        match spawn_write(cmd, args, text.as_bytes()) {
            Ok(()) => return Ok(()),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no clipboard command available")))
}

fn spawn_write(cmd: &str, args: &[&str], bytes: &[u8]) -> Result<()> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("spawn {cmd}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(bytes)?;
    }
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("{cmd} exited {status}");
    }
}

/// Open an http(s) URL in the default browser. Rejects file/javascript/data.
pub fn open_http_url(raw: &str) -> Result<()> {
    let url = parse_safe_http_url(raw)
        .ok_or_else(|| anyhow::anyhow!("only http(s) URLs can be opened"))?;
    open_path(&url)
}

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Resident set for this TUI process. Ink `/mem` is V8 heap; we report RSS.
pub fn process_mem() -> String {
    let pid = std::process::id();
    let rss = process_rss_kb(pid);
    let virt = process_virt_kb(pid);
    let mut body = format!("pid {pid}\n");
    if let Some(kb) = rss {
        body.push_str(&format!("rss  {}\n", fmt_kib(kb)));
    }
    if let Some(kb) = virt {
        body.push_str(&format!("virt {}\n", fmt_kib(kb)));
    }
    if rss.is_none() && virt.is_none() {
        body.push_str("rss unknown on this host\n");
    }
    body.trim_end().to_string()
}

fn fmt_kib(kb: u64) -> String {
    if kb >= 1024 {
        format!("{:.1} MiB", kb as f64 / 1024.0)
    } else {
        format!("{kb} KiB")
    }
}

fn process_rss_kb(pid: u32) -> Option<u64> {
    if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
        let out = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .ok()
    } else {
        None
    }
}

fn process_virt_kb(pid: u32) -> Option<u64> {
    if cfg!(target_os = "macos") || cfg!(target_os = "linux") {
        let out = Command::new("ps")
            .args(["-o", "vsz=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .ok()
    } else {
        None
    }
}

pub fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | bytes[i + 2] as u32;
        out.push(B64[((n >> 18) & 63) as usize] as char);
        out.push(B64[((n >> 12) & 63) as usize] as char);
        out.push(B64[((n >> 6) & 63) as usize] as char);
        out.push(B64[(n & 63) as usize] as char);
        i += 3;
    }
    match bytes.len() - i {
        1 => {
            let n = (bytes[i] as u32) << 16;
            out.push(B64[((n >> 18) & 63) as usize] as char);
            out.push(B64[((n >> 12) & 63) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
            out.push(B64[((n >> 18) & 63) as usize] as char);
            out.push(B64[((n >> 12) & 63) as usize] as char);
            out.push(B64[((n >> 6) & 63) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(any(test, target_os = "windows"))]
pub fn decode_base64(input: &str) -> Option<Vec<u8>> {
    let mut table = [0xffu8; 256];
    for (i, &c) in B64.iter().enumerate() {
        table[c as usize] = i as u8;
    }
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if clean.is_empty() || !clean.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    let mut i = 0;
    while i < clean.len() {
        let mut n = 0u32;
        let mut pad = 0u32;
        for k in 0..4 {
            let c = clean[i + k];
            if c == b'=' {
                pad += 1;
                continue;
            }
            let v = table[c as usize];
            if v == 0xff {
                return None;
            }
            n |= (v as u32) << (18 - 6 * k);
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
        i += 4;
    }
    Some(out)
}

/// Local clipboard PNG when `clipboard.paste` cannot see the image (SSH, remote).
pub fn read_clipboard_png() -> Option<Vec<u8>> {
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";
    let timeout = std::time::Duration::from_secs(5);
    if cfg!(target_os = "macos") {
        let temp = tempfile::Builder::new()
            .prefix("hermes-clip-")
            .suffix(".png")
            .tempfile()
            .ok()?;
        let dest = temp.path().to_path_buf();
        let path = dest.to_string_lossy().replace('\\', "/");
        if path.contains('"') {
            return None;
        }
        let dest_s = dest.to_string_lossy().to_string();
        let _ = capture_cmd("pngpaste", &[&dest_s], timeout);
        if let Ok(bytes) = std::fs::read(&dest) {
            if bytes.starts_with(PNG) {
                return Some(bytes);
            }
        }
        let script = format!(
            "try\n  set imgData to the clipboard as «class PNGf»\n  set f to open for access POSIX file \"{path}\" with write permission\n  write imgData to f\n  close access f\nend try"
        );
        let _ = capture_cmd("osascript", &["-e", &script], timeout);
        if let Ok(bytes) = std::fs::read(&dest) {
            if bytes.starts_with(PNG) {
                return Some(bytes);
            }
        }
        return None;
    }
    #[cfg(target_os = "windows")]
    {
        return read_clipboard_png_windows(timeout);
    }
    if let Some(bytes) = capture_cmd("wl-paste", &["--type", "image/png"], timeout) {
        if bytes.starts_with(PNG) {
            return Some(bytes);
        }
    }
    if let Some(bytes) = capture_cmd(
        "xclip",
        &["-selection", "clipboard", "-t", "image/png", "-o"],
        timeout,
    ) {
        if bytes.starts_with(PNG) {
            return Some(bytes);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn read_clipboard_png_windows(timeout: std::time::Duration) -> Option<Vec<u8>> {
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";
    let scripts = [
        "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; $img = [System.Windows.Forms.Clipboard]::GetImage(); if ($null -eq $img) { exit 1 }; $ms = New-Object System.IO.MemoryStream; $img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); [System.Convert]::ToBase64String($ms.ToArray())",
        "try { Add-Type -AssemblyName System.Drawing; $img = Get-Clipboard -Format Image -ErrorAction Stop; if ($null -eq $img) { exit 1 }; $ms = New-Object System.IO.MemoryStream; $img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png); [System.Convert]::ToBase64String($ms.ToArray()) } catch { exit 1 }",
    ];
    for script in scripts {
        let Some(out) = capture_cmd(
            "powershell",
            &["-NoProfile", "-NonInteractive", "-Command", script],
            timeout,
        ) else {
            continue;
        };
        let text = String::from_utf8_lossy(&out);
        if let Some(bytes) = decode_base64(text.trim()) {
            if bytes.starts_with(PNG) {
                return Some(bytes);
            }
        }
    }
    None
}

fn capture_cmd(cmd: &str, args: &[&str], timeout: std::time::Duration) -> Option<Vec<u8>> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut buf = Vec::new();
                let _ = stdout.read_to_end(&mut buf);
                if status.success() && !buf.is_empty() {
                    return Some(buf);
                }
                return None;
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
}

pub fn parse_safe_http_url(raw: &str) -> Option<String> {
    let url = raw.trim();
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    if host.is_empty() || host.contains(' ') {
        return None;
    }
    Some(url.to_string())
}

/// Open a path with the platform file opener (`open` / `xdg-open` / `explorer`).
pub fn open_path(path: &str) -> Result<()> {
    let (cmd, args): (&str, Vec<&str>) = if cfg!(target_os = "macos") {
        ("open", vec![path])
    } else if cfg!(target_os = "windows") {
        ("explorer", vec![path])
    } else {
        ("xdg-open", vec![path])
    };
    Command::new(cmd)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("open {path}"))?;
    Ok(())
}

/// Bounded `git -C <cwd> …` probe. Mirrors tui_gateway.git_probe (1.5s).
pub fn git_probe(cwd: &str, args: &[&str], timeout: std::time::Duration) -> Option<String> {
    if cwd.is_empty() || !std::path::Path::new(cwd).is_dir() {
        return None;
    }
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(cwd).args(args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::null());
    let mut child = cmd.spawn().ok()?;
    let mut stdout = child.stdout.take()?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut buf = Vec::new();
                let _ = stdout.read_to_end(&mut buf);
                if !status.success() {
                    return None;
                }
                let s = String::from_utf8_lossy(&buf).trim().to_string();
                return if s.is_empty() { None } else { Some(s) };
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
}

pub fn git_status_porcelain(cwd: &str) -> String {
    git_probe(
        cwd,
        &["status", "--porcelain", "-uall"],
        std::time::Duration::from_millis(1500),
    )
    .unwrap_or_default()
}

pub fn git_file_diff(cwd: &str, rel: &str) -> String {
    let timeout = std::time::Duration::from_millis(1500);
    if let Some(s) = git_probe(cwd, &["diff", "HEAD", "--", rel], timeout) {
        if !s.trim().is_empty() {
            return truncate_diff(&s);
        }
    }
    if let Some(s) = git_probe(cwd, &["diff", "--", rel], timeout) {
        if !s.trim().is_empty() {
            return truncate_diff(&s);
        }
    }
    let path = std::path::Path::new(cwd).join(rel);
    match std::fs::read_to_string(&path) {
        Ok(body) if !body.trim().is_empty() => {
            let mut out = String::from("(untracked / unchanged vs HEAD)\n");
            for (i, line) in body.lines().enumerate() {
                if i >= 80 {
                    out.push_str("…\n");
                    break;
                }
                out.push_str(line);
                out.push('\n');
            }
            out
        }
        _ => "(no diff for this path)".into(),
    }
}

pub fn git_restore_worktree(cwd: &str, rel: &str) -> std::result::Result<(), String> {
    git_try(
        cwd,
        &["restore", "--worktree", "--source=HEAD", "--", rel],
        std::time::Duration::from_secs(8),
    )
    .map(|_| ())
}

/// Join `rel` under `cwd` only when it cannot walk out of the workspace.
pub fn confined_worktree_path(
    cwd: &str,
    rel: &str,
) -> std::result::Result<std::path::PathBuf, String> {
    if rel.is_empty() || rel.contains('\0') {
        return Err("bad path".into());
    }
    let rel_path = std::path::Path::new(rel);
    if rel_path.is_absolute() {
        return Err("bad path".into());
    }
    for c in rel_path.components() {
        match c {
            std::path::Component::Normal(_) | std::path::Component::CurDir => {}
            _ => return Err("bad path".into()),
        }
    }
    let root = std::path::Path::new(cwd)
        .canonicalize()
        .map_err(|e| format!("workspace path: {e}"))?;
    let mut path = root.clone();
    for component in rel_path.components() {
        let std::path::Component::Normal(component) = component else {
            continue;
        };
        path.push(component);
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => return Err("symlink path".into()),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.to_string()),
        }
    }
    if path.exists() {
        let canonical = path.canonicalize().map_err(|e| e.to_string())?;
        if !canonical.starts_with(&root) {
            return Err("path escapes workspace".into());
        }
    }
    let path = root.join(rel_path);
    Ok(path)
}

pub fn read_worktree_bytes(cwd: &str, rel: &str) -> std::result::Result<Vec<u8>, String> {
    let path = confined_worktree_path(cwd, rel)?;
    std::fs::read(&path).map_err(|e| e.to_string())
}

pub fn write_worktree_bytes(cwd: &str, rel: &str, bytes: &[u8]) -> std::result::Result<(), String> {
    let mut path = confined_worktree_path(cwd, rel)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    path = confined_worktree_path(cwd, rel)?;
    std::fs::write(&path, bytes).map_err(|e| e.to_string())
}

fn truncate_diff(s: &str) -> String {
    truncate_diff_n(s, 200)
}

fn truncate_diff_n(s: &str, max_lines: usize) -> String {
    let mut out = String::new();
    for (i, line) in s.lines().enumerate() {
        if i >= max_lines {
            out.push_str("…\n");
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

pub fn git_file_patch(cwd: &str, rel: &str) -> String {
    let timeout = std::time::Duration::from_millis(1800);
    if let Some(s) = git_probe(cwd, &["diff", "-U8", "HEAD", "--", rel], timeout) {
        if !s.trim().is_empty() {
            return truncate_diff_n(&s, 400);
        }
    }
    if let Some(s) = git_probe(cwd, &["diff", "-U8", "--", rel], timeout) {
        if !s.trim().is_empty() {
            return truncate_diff_n(&s, 400);
        }
    }
    git_file_diff(cwd, rel)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirtyFile {
    pub rel: String,
    pub mark: char,
    pub check: Vec<String>,
}

pub fn parse_diff_check_notes(text: &str) -> std::collections::HashMap<String, Vec<String>> {
    let mut map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        if let Some((path, note)) = parse_check_header(line) {
            current = Some(path.clone());
            map.entry(path).or_default().push(note);
        } else if let Some(path) = &current {
            if !line.trim().is_empty() {
                map.entry(path.clone()).or_default().push(line.to_string());
            }
        }
    }
    map
}

fn parse_check_header(line: &str) -> Option<(String, String)> {
    let mut parts = line.rsplitn(3, ':');
    let msg = parts.next()?.trim();
    let line_no = parts.next()?.trim();
    let path = parts.next()?.trim();
    if path.is_empty() || !line_no.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((path.replace('\\', "/"), format!("{line_no}: {msg}")))
}

pub fn list_dirty_files(porcelain: &str, check_text: &str) -> Vec<DirtyFile> {
    let notes = parse_diff_check_notes(check_text);
    let marks = crate::fs_tree::parse_porcelain(porcelain);
    let mut rels: Vec<String> = marks.keys().cloned().collect();
    for path in notes.keys() {
        if !marks.contains_key(path) {
            rels.push(path.clone());
        }
    }
    rels.sort();
    rels.dedup();
    let mut files: Vec<DirtyFile> = rels
        .into_iter()
        .map(|rel| {
            let mark = marks.get(&rel).copied().unwrap_or('M');
            let check = notes.get(&rel).cloned().unwrap_or_default();
            DirtyFile { rel, mark, check }
        })
        .collect();
    files.sort_by(|a, b| {
        a.check
            .is_empty()
            .cmp(&b.check.is_empty())
            .then(a.rel.cmp(&b.rel))
    });
    files
}

/// stdout of `git -C cwd …`, even when the command exits non-zero
/// (`git diff --check` does that when it finds problems).
pub fn git_stdout(cwd: &str, args: &[&str], timeout: std::time::Duration) -> Option<String> {
    if cwd.is_empty() || !std::path::Path::new(cwd).is_dir() {
        return None;
    }
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(cwd).args(args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::null());
    let mut child = cmd.spawn().ok()?;
    let mut stdout = child.stdout.take()?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let mut buf = Vec::new();
                let _ = stdout.read_to_end(&mut buf);
                let s = String::from_utf8_lossy(&buf).trim().to_string();
                return if s.is_empty() { None } else { Some(s) };
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
}

pub fn porcelain_dirty_summary(porcelain: &str) -> String {
    if porcelain.is_empty() {
        return String::new();
    }
    let mut m = 0u32;
    let mut a = 0u32;
    let mut d = 0u32;
    let mut u = 0u32;
    for line in porcelain.lines() {
        let bytes = line.as_bytes();
        if bytes.len() < 2 {
            continue;
        }
        let (x, y) = (bytes[0], bytes[1]);
        if x == b'?' && y == b'?' {
            u += 1;
            continue;
        }
        if x == b'M' || y == b'M' {
            m += 1;
        }
        if x == b'A' || y == b'A' {
            a += 1;
        }
        if x == b'D' || y == b'D' {
            d += 1;
        }
    }
    let mut parts = Vec::new();
    if m > 0 {
        parts.push(format!("{m}M"));
    }
    if a > 0 {
        parts.push(format!("{a}A"));
    }
    if d > 0 {
        parts.push(format!("{d}D"));
    }
    if u > 0 {
        parts.push(format!("{u}?"));
    }
    parts.join(" ")
}

pub fn git_dirty_summary(cwd: &str) -> String {
    porcelain_dirty_summary(&git_status_porcelain(cwd))
}

pub fn git_diff_check(cwd: &str) -> String {
    let timeout = std::time::Duration::from_millis(1500);
    let mut out = String::new();
    if let Some(s) = git_probe(cwd, &["status", "-sb"], timeout) {
        out.push_str(&s);
        out.push('\n');
    }
    if let Some(s) = git_stdout(cwd, &["diff", "--check", "HEAD"], timeout) {
        out.push('\n');
        out.push_str("## diff --check\n");
        out.push_str(&s);
        out.push('\n');
    } else if let Some(s) = git_stdout(cwd, &["diff", "--check"], timeout) {
        out.push('\n');
        out.push_str("## diff --check\n");
        out.push_str(&s);
        out.push('\n');
    }
    if let Some(s) = git_probe(cwd, &["diff", "--stat", "HEAD"], timeout) {
        out.push('\n');
        out.push_str(&s);
        out.push('\n');
    }
    if let Some(s) = git_probe(cwd, &["diff", "HEAD"], timeout) {
        out.push('\n');
        for (i, line) in s.lines().enumerate() {
            if i >= 160 {
                out.push_str("…\n");
                break;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    let trimmed = out.trim();
    if trimmed.is_empty() {
        "(working tree clean — no unstaged diff vs HEAD)".into()
    } else {
        trimmed.to_string()
    }
}

pub fn git_diff_snapshot(cwd: &str) -> String {
    let timeout = std::time::Duration::from_millis(1500);
    let mut out = String::new();
    if let Some(s) = git_probe(cwd, &["status", "-sb"], timeout) {
        out.push_str(&s);
        out.push('\n');
    }
    if let Some(s) = git_probe(cwd, &["diff", "--stat", "HEAD"], timeout) {
        out.push('\n');
        out.push_str(&s);
        out.push('\n');
    }
    if let Some(s) = git_probe(cwd, &["diff", "HEAD"], timeout) {
        out.push('\n');
        for (i, line) in s.lines().enumerate() {
            if i >= 160 {
                out.push_str("…\n");
                break;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    let trimmed = out.trim();
    if trimmed.is_empty() {
        "(working tree clean — no unstaged diff vs HEAD)".into()
    } else {
        trimmed.to_string()
    }
}

pub fn probe_git_repo_branch(cwd: &str) -> (Option<String>, Option<String>) {
    let timeout = std::time::Duration::from_millis(1500);
    let repo = git_probe(cwd, &["rev-parse", "--show-toplevel"], timeout).and_then(|s| {
        std::path::Path::new(&s)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
    });
    let branch = git_probe(cwd, &["branch", "--show-current"], timeout)
        .or_else(|| git_probe(cwd, &["rev-parse", "--short", "HEAD"], timeout));
    (repo, branch)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitBranch {
    pub name: String,
    pub current: bool,
    pub worktree: Option<String>,
}

/// Parse `git for-each-ref --format='%(HEAD)|%(refname:short)|%(worktreepath)'`.
pub fn parse_branch_refs(text: &str) -> Vec<GitBranch> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, '|');
        let head = parts.next().unwrap_or("");
        let name = parts.next().unwrap_or("").trim();
        if name.is_empty() {
            continue;
        }
        let wt = parts.next().unwrap_or("").trim();
        out.push(GitBranch {
            name: name.to_string(),
            current: head.contains('*'),
            worktree: if wt.is_empty() {
                None
            } else {
                Some(wt.to_string())
            },
        });
    }
    out
}

pub fn list_git_branches(cwd: &str) -> Vec<GitBranch> {
    let timeout = std::time::Duration::from_millis(1500);
    let raw = git_probe(
        cwd,
        &[
            "for-each-ref",
            "--format=%(HEAD)|%(refname:short)|%(worktreepath)",
            "refs/heads",
        ],
        timeout,
    );
    parse_branch_refs(&raw.unwrap_or_default())
}

pub fn switch_git_branch(cwd: &str, name: &str) -> std::result::Result<(), String> {
    git_try(
        cwd,
        &["switch", "--", name],
        std::time::Duration::from_secs(8),
    )
    .map(|_| ())
}

fn git_try(
    cwd: &str,
    args: &[&str],
    timeout: std::time::Duration,
) -> std::result::Result<String, String> {
    if cwd.is_empty() || !std::path::Path::new(cwd).is_dir() {
        return Err("not a directory".into());
    }
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(cwd).args(args);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = Vec::new();
                let mut err = Vec::new();
                if let Some(mut s) = stdout.take() {
                    let _ = s.read_to_end(&mut out);
                }
                if let Some(mut s) = stderr.take() {
                    let _ = s.read_to_end(&mut err);
                }
                let out_s = String::from_utf8_lossy(&out).trim().to_string();
                let err_s = String::from_utf8_lossy(&err).trim().to_string();
                if status.success() {
                    return Ok(out_s);
                }
                return Err(if err_s.is_empty() { out_s } else { err_s });
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("git timed out".into());
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(e) => return Err(e.to_string()),
        }
    }
}

pub fn same_path(a: &str, b: &str) -> bool {
    let pa = std::path::Path::new(a);
    let pb = std::path::Path::new(b);
    match (pa.canonicalize(), pb.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a.trim_end_matches('/') == b.trim_end_matches('/'),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_probe_empty_cwd_is_none() {
        assert!(git_probe("", &["status"], std::time::Duration::from_millis(100)).is_none());
    }

    #[test]
    fn git_probe_missing_dir_is_none() {
        assert!(git_probe(
            "/definitely/not/a/real/hermes-tui-path",
            &["status"],
            std::time::Duration::from_millis(100)
        )
        .is_none());
    }

    #[test]
    fn git_dirty_summary_counts_marks() {
        assert_eq!(porcelain_dirty_summary(""), "");
        assert_eq!(
            porcelain_dirty_summary(" M src/app.rs\nA  new.rs\n D gone.rs\n?? scratch.rs\n"),
            "1M 1A 1D 1?"
        );
        assert_eq!(git_dirty_summary(""), "");
    }

    #[test]
    fn parse_diff_check_and_dirty_files() {
        let check = "src/app.rs:12: trailing whitespace.\n+    foo   \nsrc/app.rs:40: leftover conflict marker\n<<<<<<< HEAD\n";
        let notes = parse_diff_check_notes(check);
        assert_eq!(notes["src/app.rs"].len(), 4);
        assert!(notes["src/app.rs"][0].contains("trailing whitespace"));
        let files = list_dirty_files(" M src/app.rs\n?? scratch.rs\n", check);
        assert_eq!(files[0].rel, "src/app.rs");
        assert!(!files[0].check.is_empty());
        assert_eq!(files[1].rel, "scratch.rs");
        assert_eq!(files[1].mark, '?');
    }

    #[test]
    fn parse_branch_refs_marks_current_and_worktree() {
        let text = "*|main|/var/tmp/hermes-tui-src\n |feat/picker|\n |other|/tmp/other\n";
        let rows = parse_branch_refs(text);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].name, "main");
        assert!(rows[0].current);
        assert_eq!(rows[0].worktree.as_deref(), Some("/var/tmp/hermes-tui-src"));
        assert!(!rows[1].current);
        assert!(rows[1].worktree.is_none());
        assert_eq!(rows[2].worktree.as_deref(), Some("/tmp/other"));
    }

    #[test]
    fn worktree_bytes_reject_absolute_and_roundtrip() {
        assert!(read_worktree_bytes(".", "/etc/passwd").is_err());
        assert!(write_worktree_bytes(".", "/tmp/x", b"no").is_err());
        assert!(confined_worktree_path(".", "../secret").is_err());
        assert!(confined_worktree_path(".", "foo/../x").is_err());
        assert!(write_worktree_bytes(".", "../outside.txt", b"no").is_err());
        let dir = std::env::temp_dir().join(format!("hermes-tui-wt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let cwd = dir.to_string_lossy().to_string();
        write_worktree_bytes(&cwd, "n.txt", b"ok").unwrap();
        assert_eq!(read_worktree_bytes(&cwd, "n.txt").unwrap(), b"ok");
        write_worktree_bytes(&cwd, "sub/a.txt", b"in").unwrap();
        assert_eq!(read_worktree_bytes(&cwd, "sub/a.txt").unwrap(), b"in");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn worktree_bytes_reject_symlinked_parent() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("hermes-tui-root-{}", uuid::Uuid::new_v4()));
        let outside =
            std::env::temp_dir().join(format!("hermes-tui-outside-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        symlink(&outside, root.join("linked")).unwrap();
        let cwd = root.to_string_lossy();
        assert!(write_worktree_bytes(&cwd, "linked/file", b"no").is_err());
        assert!(!outside.join("file").exists());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn process_mem_mentions_pid() {
        let body = process_mem();
        assert!(body.contains(&format!("pid {}", std::process::id())));
    }

    #[test]
    fn encode_base64_pads() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"hi"), "aGk=");
        assert_eq!(encode_base64(&[0, 0, 0]), "AAAA");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(decode_base64("aGk=").as_deref(), Some(b"hi".as_slice()));
        assert_eq!(decode_base64("Zg==").as_deref(), Some(b"f".as_slice()));
        assert_eq!(
            decode_base64(&encode_base64(b"png\nbytes")),
            Some(b"png\nbytes".to_vec())
        );
    }

    #[test]
    fn parse_safe_http_url_rejects_file_and_js() {
        assert!(parse_safe_http_url("https://example.com/oauth?x=1").is_some());
        assert!(parse_safe_http_url("http://127.0.0.1:9/callback").is_some());
        assert!(parse_safe_http_url("file:///etc/passwd").is_none());
        assert!(parse_safe_http_url("javascript:alert(1)").is_none());
        assert!(parse_safe_http_url("https://").is_none());
    }
}
