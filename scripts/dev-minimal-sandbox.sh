#!/usr/bin/env bash
# Preserve this entrypoint's separate identity without duplicating sandbox lifecycle.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dev-sandbox.sh"
