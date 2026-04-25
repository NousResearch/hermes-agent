#!/bin/bash
source /home/ubuntu/.openclaw/.env
cd /home/ubuntu/.openclaw/scripts/pipeline/technical
python3 real_time.py 2>&1 | grep -v "^Response Code\|^WARNING\|^/home"