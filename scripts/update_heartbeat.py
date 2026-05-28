#!/usr/bin/env python3
"""Heartbeat updater — touch this periodically during dev sprints.
Usage: python3 update_heartbeat.py
       or source it in a cron job every 2-3 minutes.
"""
import os, time

HEARTBEAT = os.path.expanduser("~/.hermes/agent_heartbeat")

def update():
    os.makedirs(os.path.dirname(HEARTBEAT), exist_ok=True)
    with open(HEARTBEAT, "w") as f:
        f.write(f"alive {int(time.time())}")

if __name__ == "__main__":
    update()
    print(f"Heartbeat updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")