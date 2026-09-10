#!/bin/sh
set -eu

repository=${1:?repository is required}
export HERMES_PR_FEEDBACK_REPOSITORY="$repository"
exec /Users/mikedemott/.hermes/scripts/github-pr-feedback-scan.sh
