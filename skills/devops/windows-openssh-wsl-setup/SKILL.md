---
name: windows-openssh-wsl-setup
description: Set up OpenSSH Server on Windows 10/11 and configure WSL Ubuntu autostart at login. Use when remotely managing a Windows machine via SSH, or setting up WSL as an always-on Linux environment on a Windows host.
tags: [windows, ssh, wsl, openssh, tailscale, devops]
---

# Windows OpenSSH + WSL Autostart Setup

## Trigger
Use when:
- Need SSH access into a Windows 10/11 machine
- Want WSL to start automatically at boot/login
- Setting up a Windows laptop as a remote execution node

## Key Findings (from trial and error)

### WSL + Tailscale
WSL2 on Windows automatically shares the Windows host Tailscale network.
No separate Tailscale install needed inside WSL — Tailscale IPs are directly reachable from WSL.
Test with: `ping <tailscale-ip>` from inside WSL.

### PowerShell pitfall
Users often copy terminal output including "PS C:\Users\...>" prompts into a new PowerShell session.
This causes cascade parse errors. Always instruct: copy commands only, one line at a time.

### Elevation required
All OpenSSH and service commands require an elevated PowerShell session.
Instruct: Windows key -> type "PowerShell" -> right-click -> "Run as administrator"

## Steps

### 1. Install and Start OpenSSH Server (admin PowerShell)
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```
Run each line separately. `Add-WindowsCapability` can take 30-60 seconds.

### 2. Enable Password Authentication (for initial access)
```powershell
(Get-Content "C:\ProgramData\ssh\sshd_config") -replace '#PasswordAuthentication yes','PasswordAuthentication yes' | Set-Content "C:\ProgramData\ssh\sshd_config"
Restart-Service sshd
```

### 3. Add SSH Public Key (passwordless access)
```powershell
$key = "<paste public key here>"
$authFile = "C:\Users\sander\.ssh\authorized_keys"
New-Item -ItemType Directory -Force -Path "C:\Users\sander\.ssh"
Add-Content -Path $authFile -Value $key
```
Then fix permissions (Windows OpenSSH requires strict ACL):
```powershell
icacls "C:\Users\sander\.ssh\authorized_keys" /inheritance:r /grant "sander:(R)" /grant "SYSTEM:(R)"
```

### 4. WSL Autostart at Login (admin PowerShell)
```powershell
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu"
$trigger = New-ScheduledTaskTrigger -AtLogon
Register-ScheduledTask -TaskName "WSL Autostart" -Action $action -Trigger $trigger -RunLevel Highest
```

### 5. Test SSH from remote
```bash
ssh -o StrictHostKeyChecking=no sander@<tailscale-ip> "echo SSH_OK && whoami"
```

## Pitfalls
- authorized_keys permissions: Windows OpenSSH ignores keys if ACL is wrong. Always run icacls after creating the file.
- Run PowerShell as Administrator or ALL service/capability commands fail silently or with "elevation required".
- Do NOT paste multi-line output back into PowerShell — only paste the command itself.
- If sshd won't start after Add-WindowsCapability, reboot once then retry Start-Service.
- WSL default distro name matters in -Argument: check with `wsl --list` if Ubuntu doesn't start.

## Verification
```bash
# From Pluto/Hermes (WSL side):
ssh sander@100.116.48.49 "echo OK"
# Should return: OK
```
