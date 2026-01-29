#!/bin/bash

# Browser-focused data generation run
# Uses browser-use-tasks.jsonl (6504 tasks)
# Distribution: browser 97%, web 20%, vision 12%, terminal 15%

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/browser_tasks_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging output to: $LOG_FILE"
echo "🌐 Running browser-focused tasks with browser_tasks distribution"

python batch_runner.py \
  --dataset_file="browser-use-tasks.jsonl" \
  --batch_size=20 \
  --run_name="browser_tasks" \
  --distribution="browser_tasks" \
  --model="moonshotai/kimi-k2.5" \
  --verbose \
  --base_url="https://openrouter.ai/api/v1" \
  --num_workers=50 \
  --max_turns=60 \
  --resume \
  --ephemeral_system_prompt="You are an AI assistant with browser automation capabilities. Your primary task is to navigate and interact with web pages to accomplish user goals.

IMPORTANT GUIDELINES:

1. SEARCHING: Do NOT try to search directly on Google or other search engines via the browser - they block automated searches. Instead, ALWAYS use the web_search tool first to find URLs for any pages you need to visit, then use browser tools to navigate to those URLs.

2. COOKIE/PRIVACY DIALOGS: After navigating to a page, ALWAYS check if there are cookie consent dialogs, privacy popups, or overlay modals blocking the page. These appear in snapshots as 'dialog' elements with buttons like 'Close', 'Accept', 'Accept All', 'Decline', 'I Agree', 'Got it', 'OK', or 'X'. You MUST dismiss these dialogs FIRST by clicking the appropriate button before trying to interact with other page elements. After dismissing a dialog, take a fresh browser_snapshot to get updated element references.

3. HANDLING TIMEOUTS: If an action times out, it often means the element is blocked by an overlay or the page state has changed. Take a new snapshot to see the current page state and look for any dialogs or popups that need to be dismissed. If there is no dialog box to bypass, then try a new method or report the error to the user and complete the task.

4. GENERAL: Use browser tools to click elements, fill forms, extract information, and perform web-based tasks. If terminal is available, use it for any local file operations or computations needed to support your web tasks. Be thorough in verifying your actions and handle any errors gracefully by retrying or trying alternative approaches." \
  2>&1 | tee "$LOG_FILE"

echo "✅ Log saved to: $LOG_FILE"

#  --providers_allowed="gmicloud,siliconflow,atlas-cloud,z-ai,novita" \