#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin=""
for candidate in "$repo_root/.venv/bin/python" "$repo_root/venv/bin/python" "$(command -v python3 || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    python_bin="$candidate"
    break
  fi
done
if [[ -z "$python_bin" ]]; then
  echo "Python runtime not found" >&2
  exit 1
fi

no_systemd=false
bootstrap_args=(install --repo-root "$repo_root")
requested_hermes_root=""
requested_runtime_home=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-systemd)
      no_systemd=true
      shift
      ;;
    --hermes-root|--runtime-home)
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 2; }
      if [[ "$1" == "--hermes-root" ]]; then
        requested_hermes_root="$2"
      else
        requested_runtime_home="$2"
      fi
      bootstrap_args+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Usage: $0 [--no-systemd] [--hermes-root PATH] [--runtime-home PATH]" >&2
      exit 2
      ;;
  esac
done

cd "$repo_root"
"$python_bin" -m hegi.bootstrap "${bootstrap_args[@]}"
if [[ -n "$requested_runtime_home" ]]; then
  runtime_home="$requested_runtime_home"
elif [[ -n "$requested_hermes_root" ]]; then
  runtime_home="$("$python_bin" -m hegi.bootstrap locate-home --hermes-root "$requested_hermes_root")"
else
  runtime_home="$("$python_bin" -m hegi.bootstrap locate-home)"
fi
mkdir -p "$runtime_home/hegi/archive"

restart_recovery="none"
if [[ "$no_systemd" == false ]] && command -v systemctl >/dev/null 2>&1 \
  && systemctl --user show-environment >/dev/null 2>&1; then
  unit_dir="$HOME/.config/systemd/user"
  unit_path="$unit_dir/hegi.service"
  mkdir -p "$unit_dir"
  temporary="$unit_path.tmp"
  {
    echo "[Unit]"
    echo "Description=HEGI v2 AI Research Secretary"
    echo "After=network-online.target"
    echo "Wants=network-online.target"
    echo
    echo "[Service]"
    printf 'Type=simple\n'
    printf 'WorkingDirectory=%s\n' "$repo_root"
    printf 'Environment=HERMES_HOME=%s\n' "$runtime_home"
    printf 'ExecStart=%s -m hegi daemon --send\n' "$python_bin"
    printf 'Restart=on-failure\n'
    printf 'RestartSec=5\n'
    printf 'TimeoutStopSec=20\n'
    echo
    echo "[Install]"
    echo "WantedBy=default.target"
  } >"$temporary"
  chmod 600 "$temporary"
  mv "$temporary" "$unit_path"
  systemctl --user daemon-reload
  systemctl --user enable hegi.service >/dev/null
  restart_recovery="systemd"
  echo "HEGI restart recovery enabled: $unit_path"
elif [[ "$no_systemd" == false && -n "${WSL_DISTRO_NAME:-}" ]] \
  && command -v powershell.exe >/dev/null 2>&1; then
  wsl_start_command="HERMES_HOME='$runtime_home' '$repo_root/hegi/scripts/start.sh' --send"
  distro_b64="$(printf '%s' "$WSL_DISTRO_NAME" | base64 -w0)"
  command_b64="$(printf '%s' "$wsl_start_command" | base64 -w0)"
  recovery_script="$runtime_home/hegi/install-recovery.ps1"
  {
    printf '$distroName = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("%s"))\n' "$distro_b64"
    printf '$startCommand = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("%s"))\n' "$command_b64"
    cat <<'POWERSHELL'
    $arguments = "-d `"" + $distroName + "`" -- bash -lc `"" +
      $startCommand + "`""
    $action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wsl.exe" -Argument $arguments
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    try {
      Register-ScheduledTask -TaskName "Hermes HEGI v2" -Action $action -Trigger $trigger `
        -Description "Start HEGI v2 in WSL after Windows logon" -Force -ErrorAction Stop | Out-Null
      "windows-task"
    } catch {
      $startup = [Environment]::GetFolderPath("Startup")
      $launcher = Join-Path $startup "Hermes-HEGI-v2.cmd"
      $line = "@echo off`r`nstart `"`" /min wsl.exe " + $arguments + "`r`n"
      [IO.File]::WriteAllText($launcher, $line, [Text.UTF8Encoding]::new($false))
      "windows-startup"
    }
POWERSHELL
  } >"$recovery_script"
  recovery_method="$(powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$(wslpath -w "$recovery_script")")"
  rm -f "$recovery_script"
  restart_recovery="$(printf '%s' "$recovery_method" | tr -d '\r\n')"
  echo "HEGI restart recovery enabled: $restart_recovery / Hermes HEGI v2"
else
  echo "HEGI systemd unit skipped; start.sh will use the supervised PID/lock fallback."
fi

HERMES_HOME="$runtime_home" "$python_bin" -m hegi doctor
HERMES_HOME="$runtime_home" "$repo_root/hegi/scripts/restart_gateway.sh"
