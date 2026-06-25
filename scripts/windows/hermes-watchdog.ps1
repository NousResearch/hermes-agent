# Hermes Watchdog — background health monitor
# Runs in the background, checks port 8642 every 5 minutes, auto-recovers if down.
# Launched by startup-hermes.cmd at boot time.

$HERMES_HOME = "C:\Users\aliyf\AppData\Local\hermes"
$LOGFILE = "$HERMES_HOME\logs\watchdog.log"
$CHECK_INTERVAL_SECS = 300  # 5 minutes

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Watchdog started (interval=${CHECK_INTERVAL_SECS}s)" | Out-File -Append -Encoding utf8 $LOGFILE

while ($true) {
    Start-Sleep -Seconds $CHECK_INTERVAL_SECS

    try {
        # Check if port 8642 is listening
        $listening = Get-NetTCPConnection -LocalPort 8642 -State Listen -ErrorAction SilentlyContinue
        if ($listening) {
            # Gateway is alive — nothing to do
            continue
        }

        # Gateway is DOWN — start recovery
        $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Output "[$timestamp] HEALTH CHECK FAILED: Port 8642 not listening! Starting recovery..." | Out-File -Append -Encoding utf8 $LOGFILE

        # Step 1: Kill zombie processes
        $hermesProcs = Get-Process -Name "hermes" -ErrorAction SilentlyContinue
        if ($hermesProcs) {
            Stop-Process -Name "hermes" -Force -ErrorAction SilentlyContinue
            Write-Output "[$timestamp] Killed zombie hermes.exe" | Out-File -Append -Encoding utf8 $LOGFILE
        }

        # Step 2: Clean lock files
        Remove-Item "$HERMES_HOME\gateway.lock" -Force -ErrorAction SilentlyContinue
        Remove-Item "$HERMES_HOME\gateway.pid" -Force -ErrorAction SilentlyContinue

        # Step 3: Restart Gateway
        $env:API_SERVER_ENABLED = "true"
        $proc = Start-Process -FilePath "C:\Users\aliyf\hermes-agent\venv\Scripts\hermes.exe" `
            -ArgumentList "gateway","run","--replace" `
            -WorkingDirectory "C:\Users\aliyf\hermes-agent" `
            -WindowStyle Minimized `
            -PassThru

        Write-Output "[$timestamp] Launched new gateway (PID $($proc.Id)), waiting for port 8642..." | Out-File -Append -Encoding utf8 $LOGFILE

        # Step 4: Wait for port 8642 (up to 60 seconds)
        $ready = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 2
            $listening = Get-NetTCPConnection -LocalPort 8642 -State Listen -ErrorAction SilentlyContinue
            if ($listening) {
                $ready = $true
                break
            }
        }

        $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        if ($ready) {
            Write-Output "[$timestamp] Auto-recovery SUCCESS: Gateway listening on 8642" | Out-File -Append -Encoding utf8 $LOGFILE

            # Step 5: Check Web UI
            $webui = Get-NetTCPConnection -LocalPort 8648 -State Listen -ErrorAction SilentlyContinue
            if (-not $webui) {
                Start-Process -FilePath "C:\nodejs\node_global\hermes-web-ui.cmd" `
                    -ArgumentList "start","--port","8648" `
                    -WindowStyle Minimized
                Write-Output "[$timestamp] Web UI restarted" | Out-File -Append -Encoding utf8 $LOGFILE
            }
        } else {
            Write-Output "[$timestamp] Auto-recovery FAILED: Gateway did not start within 60s" | Out-File -Append -Encoding utf8 $LOGFILE
        }

    } catch {
        $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Output "[$timestamp] Watchdog error: $_" | Out-File -Append -Encoding utf8 $LOGFILE
    }
}
