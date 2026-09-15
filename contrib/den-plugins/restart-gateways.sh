#!/bin/bash
# Restart both Hermes gateways via launchd KeepAlive, detached from the calling gateway.
sleep 3
R=$(launchctl list | awk '/ai.hermes.gateway-reviewer$/{print $1}')
D=$(launchctl list | awk '/ai.hermes.gateway$/{print $1}')
echo "$(date '+%H:%M:%S') reviewer=$R default=$D" >> /tmp/gw-restart.log
[ -n "$R" ] && [ "$R" != "-" ] && kill -TERM "$R"
[ -n "$D" ] && [ "$D" != "-" ] && kill -TERM "$D"
sleep 25
launchctl list | grep ai.hermes.gateway >> /tmp/gw-restart.log
