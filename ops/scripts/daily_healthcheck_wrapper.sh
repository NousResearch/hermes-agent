#!/bin/bash
# Hermes Daily Healthcheck wrapper — host cron entry point.
# Runs healthcheck script on host (full visibility into systemctl/docker/ss).
# On non-zero exit, pipes alert text into hermes container and sends via feishu.
#
# Schedule: host crontab `0 9 * * *` (replaces hermes cron job 0df6b63a25cb).
set -u

SCRIPT=/home/admin/.hermes/scripts/daily_healthcheck.sh
ALERT=/home/admin/.hermes/logs/healthcheck_alert.txt
WRAPPER_LOG=/home/admin/.hermes/logs/healthcheck_wrapper.log
TS=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p "$(dirname "$WRAPPER_LOG")"

echo "[$TS] healthcheck_wrapper start" >>"$WRAPPER_LOG"

# Run the healthcheck. Exit code 0 = silent success, non-zero = alerts.
bash "$SCRIPT" >>"$WRAPPER_LOG" 2>&1
RC=$?

if [ "$RC" -eq 0 ]; then
  echo "[$TS] all green, no notification sent" >>"$WRAPPER_LOG"
  exit 0
fi

# Alert path. Read alert file, prefix header, send via container's hermes CLI.
if [ ! -s "$ALERT" ]; then
  echo "[$TS] exit=$RC but ALERT file empty/missing — sending generic notice" >>"$WRAPPER_LOG"
  MSG="⚠ Hermes 每日健康检查 exit=$RC，但 alert 文件 $ALERT 为空或缺失。请查看 $WRAPPER_LOG"
else
  ALERT_BODY=$(cat "$ALERT")
  ALERT_COUNT=$(wc -l <"$ALERT")
  MSG="⚠ Hermes 每日健康检查发现 ${ALERT_COUNT} 项风险（$TS）：

${ALERT_BODY}

详情见 ${WRAPPER_LOG}"
fi

# Send via hermes CLI in container. The 'hermes send' command without -t uses
# the channel configured in agent_directory_default in config.yaml.
# Pipe through stdin so multi-line messages don't get mangled by shell.
if echo "$MSG" | docker exec -i hermes /opt/hermes/.venv/bin/hermes send -t feishu 2>>"$WRAPPER_LOG"; then
  echo "[$TS] alert sent ($RC alerts)" >>"$WRAPPER_LOG"
else
  echo "[$TS] FAILED to send alert via docker exec" >>"$WRAPPER_LOG"
fi

exit "$RC"