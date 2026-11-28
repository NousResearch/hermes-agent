//! Installed operating-system speech services. No shell interprets message text:
//! the fixed native-service invocation receives UTF-8 content through stdin.
use super::{Preferences, Voice};
use std::{
    cell::RefCell,
    io::Write,
    path::PathBuf,
    process::{Child, Command, Stdio},
};

#[cfg(target_os = "windows")]
fn program() -> Option<PathBuf> {
    let root = std::env::var_os("SystemRoot")?;
    let path = PathBuf::from(root).join("System32/WindowsPowerShell/v1.0/powershell.exe");
    path.is_file().then_some(path)
}
#[cfg(target_os = "linux")]
fn program() -> Option<PathBuf> {
    [
        "/usr/bin/espeak-ng",
        "/usr/local/bin/espeak-ng",
        "/usr/bin/espeak",
        "/usr/local/bin/espeak",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| path.is_file())
}
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn program() -> Option<PathBuf> {
    None
}

#[cfg(target_os = "windows")]
fn powershell(path: PathBuf, script: &str) -> Command {
    use std::os::windows::process::CommandExt;
    let mut command = Command::new(path);
    command
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ])
        .creation_flags(0x0800_0000);
    command
}

pub(super) fn voices() -> Vec<Voice> {
    let Some(path) = program() else {
        return Vec::new();
    };
    #[cfg(target_os = "windows")]
    {
        let script = r#"$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $v=@($s.GetInstalledVoices() | Where-Object Enabled | ForEach-Object { @{ id=$_.VoiceInfo.Name; name=($_.VoiceInfo.Name+' ('+$_.VoiceInfo.Culture.Name+')') } }); ConvertTo-Json -Compress -InputObject $v; $s.Dispose()"#;
        let Ok(output) = powershell(path, script)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
        else {
            return Vec::new();
        };
        if !output.status.success() {
            return Vec::new();
        }
        #[derive(serde::Deserialize)]
        struct Installed {
            id: String,
            name: String,
        }
        let Ok(installed) = serde_json::from_slice::<Vec<Installed>>(&output.stdout) else {
            return Vec::new();
        };
        if installed.is_empty() {
            return Vec::new();
        }
        let mut voices = vec![Voice {
            id: String::new(),
            name: "System voice".into(),
        }];
        voices.extend(installed.into_iter().map(|v| Voice {
            id: v.id,
            name: v.name,
        }));
        voices
    }
    #[cfg(target_os = "linux")]
    {
        let Ok(output) = Command::new(path)
            .arg("--voices")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
        else {
            return Vec::new();
        };
        if !output.status.success() {
            return Vec::new();
        }
        let mut voices = vec![Voice {
            id: String::new(),
            name: "System voice".into(),
        }];
        for line in String::from_utf8_lossy(&output.stdout).lines().skip(1) {
            let columns: Vec<_> = line.split_whitespace().collect();
            if columns.len() >= 5 {
                voices.push(Voice {
                    id: columns[4].into(),
                    name: format!("{} ({})", columns[3].replace('_', " "), columns[1]),
                });
            }
        }
        voices.dedup_by(|a, b| a.id == b.id);
        voices
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = path;
        Vec::new()
    }
}

pub(super) struct Output {
    child: RefCell<Option<Child>>,
}
impl Output {
    pub(super) fn new() -> Self {
        Self {
            child: RefCell::new(None),
        }
    }
    pub(super) fn speak(&self, text: &str, preferences: &Preferences) -> Result<(), String> {
        self.stop();
        if text.trim().is_empty() {
            return Err("This message has no text to read.".into());
        }
        if text.len() > 128 * 1024 {
            return Err("This message exceeds the 128 KiB speech limit.".into());
        }
        let path = program().ok_or("The installed system speech service is unavailable")?;
        #[cfg(target_os = "windows")]
        let (mut command, input) = {
            let script = r#"$ErrorActionPreference='Stop'; [Console]::InputEncoding=[System.Text.UTF8Encoding]::new(); Add-Type -AssemblyName System.Speech; $p=([Console]::In.ReadToEnd() | ConvertFrom-Json); $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; try { if ($p.voice.Length -gt 0) { $s.SelectVoice($p.voice) }; $s.Rate=[int]$p.rate; $s.Speak([string]$p.text) } finally { $s.Dispose() }"#;
            let rate = ((preferences.speed.log2() * 5.0).round() as i32).clamp(-10, 10);
            let input = serde_json::to_vec(
                &serde_json::json!({"text":text,"voice":preferences.voice_id,"rate":rate}),
            )
            .map_err(|e| e.to_string())?;
            (powershell(path, script), input)
        };
        #[cfg(target_os = "linux")]
        let (mut command, input) = {
            let mut command = Command::new(path);
            command.args([
                "--stdin",
                "-s",
                &(175.0 * preferences.speed).round().to_string(),
            ]);
            if !preferences.voice_id.is_empty() {
                command.arg("-v").arg(&preferences.voice_id);
            }
            (command, text.as_bytes().to_vec())
        };
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let (mut command, input) = {
            let _ = preferences;
            (Command::new(path), text.as_bytes().to_vec())
        };
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Could not start system speech: {e}"))?;
        if let Some(mut stdin) = child.stdin.take() {
            std::thread::spawn(move || {
                let _ = stdin.write_all(&input);
            });
        }
        self.child.replace(Some(child));
        Ok(())
    }
    pub(super) fn speaking(&self) -> Result<bool, String> {
        let mut current = self.child.borrow_mut();
        let Some(child) = current.as_mut() else {
            return Ok(false);
        };
        match child.try_wait() {
            Ok(None) => Ok(true),
            Ok(Some(status)) => {
                current.take();
                if status.success() {
                    Ok(false)
                } else {
                    Err("The system speech service could not read this message. Check the installed voice and audio output.".into())
                }
            }
            Err(error) => {
                if let Some(mut child) = current.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                Err(format!("Could not read system speech state: {error}"))
            }
        }
    }
    pub(super) fn stop(&self) {
        if let Some(mut child) = self.child.borrow_mut().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        self.stop();
    }
}
