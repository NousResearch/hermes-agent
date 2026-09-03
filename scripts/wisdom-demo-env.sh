#!/usr/bin/env bash
# Pin every Hermes surface in this shell to the current source worktree.
#
# Source it to configure the current shell:
#   source scripts/wisdom-demo-env.sh
#
# Execute it to open a configured interactive subshell:
#   scripts/wisdom-demo-env.sh
#
# Or run one command without changing the caller's environment:
#   scripts/wisdom-demo-env.sh -- hermes wisdom status

_wisdom_demo_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/wisdom-demo-env.sh
source "${_wisdom_demo_script_dir}/lib/wisdom-demo-env.sh"
unset _wisdom_demo_script_dir
if ! wisdom_demo_export_environment; then
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 1
  fi
  exit 1
fi

if [ "${HERMES_WISDOM_QUIET:-0}" != "1" ]; then
  echo "Collective Wisdom demo environment"
  echo "  source: ${HERMES_WISDOM_REPO}"
  echo "  python: ${HERMES_WISDOM_PYTHON}"
  if [ -n "${HERMES_HOME:-}" ]; then
    echo "  profile home: ${HERMES_HOME}"
  else
    echo "  profile home: default Hermes profile"
  fi
  echo "  hermes: $(command -v hermes)"
fi

unset -f \
  wisdom_demo_repo_root \
  wisdom_demo_pick_python \
  wisdom_demo_prepend_colon_path \
  wisdom_demo_export_environment 2>/dev/null || true

if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  return 0
fi

if [ "${1:-}" = "--" ]; then
  shift
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

echo "Opening a demo shell. Exit it to return to your previous environment."
exec "${SHELL:-/bin/bash}" -i
