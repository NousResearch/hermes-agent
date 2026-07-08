#!/bin/bash
gh pr create --title "fix(cli): preserve chat -q answer by gating exit-summary screen clear (#60926)" --body "Salvage of #53025. 
Fixes #60926 where running \`hermes chat -q\` immediately erased the output because \`_print_exit_summary()\` unconditionally cleared the terminal.
- Modified \`_print_exit_summary\` to accept \`clear_screen\` argument.
- Gate the clear command in single-query mode.
- Updated \`FakeCLI\` in regression tests to match the new signature."
