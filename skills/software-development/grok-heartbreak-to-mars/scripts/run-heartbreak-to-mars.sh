#!/bin/bash
# run-heartbreak-to-mars.sh - Easy invocation for grok-heartbreak-to-mars skill
set -e
TOPIC="${1:-the type of heartbreak that gets us to mars}"
echo "Running grok-heartbreak-to-mars for: $TOPIC"
hermes chat -q "Using grok-heartbreak-to-mars and grok-xai-oauth, generate original lyrics, a short story, and a visual prompt for $TOPIC in Olivia Rodrigo's raw confessional style. Include space puns, ambition as revenge, and the launch to Mars. Load all Grok skills."
echo "Done. Results should be epic and interstellar."
