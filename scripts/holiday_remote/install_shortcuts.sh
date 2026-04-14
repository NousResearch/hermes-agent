#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SNIPPET_BEGIN="# >>> holiday-remote-shortcuts >>>"
SNIPPET_END="# <<< holiday-remote-shortcuts <<<"
SNIPPET_CONTENT=$(cat <<EOF
$SNIPPET_BEGIN
alias hr='cd $REPO_ROOT && ./scripts/holiday_remote/remote_ops.sh'
alias htm='cd $REPO_ROOT && ./scripts/holiday_remote/tmux_bootstrap.sh holiday $REPO_ROOT'
$SNIPPET_END
EOF
)

install_into_rc() {
  local rc_file="$1"
  touch "$rc_file"

  if grep -q "$SNIPPET_BEGIN" "$rc_file"; then
    echo "shortcuts already installed in $rc_file"
    return 0
  fi

  printf "\n%s\n" "$SNIPPET_CONTENT" >> "$rc_file"
  echo "installed shortcuts in $rc_file"
}

install_into_rc "$HOME/.bashrc"
install_into_rc "$HOME/.zshrc"

echo "Done. Open a new shell or run: source ~/.bashrc"
