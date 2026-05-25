#!/bin/bash
MODE="${HERMES_MODE:-gateway}"

case "$MODE" in
  gateway)
    if pgrep -f 'hermes gateway' > /dev/null; then
      exit 0
    else
      exit 1
    fi
    ;;
  dashboard)
    if curl -sf http://localhost:9119/ > /dev/null 2>&1; then
      exit 0
    else
      exit 1
    fi
    ;;
  *)
    exit 1
    ;;
esac
