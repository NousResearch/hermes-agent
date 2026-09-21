#!/bin/bash
# GIT_ASKPASS helper for cron jobs — reads GH_TOKEN from env and outputs it
# Used by vault_daily_pull.sh and wiki_daily_sync.sh
echo "$GH_TOKEN"
