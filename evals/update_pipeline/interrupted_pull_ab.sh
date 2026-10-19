#!/usr/bin/env bash
# Live A/B for the interrupted-pull repair (hermes_cli/_early_recovery.py::restore_interrupted_pull).
#
#   evals/update_pipeline/interrupted_pull_ab.sh <repo> <installed-sha> <label> [python]
#
# Builds a disposable origin + install at <installed-sha>; origin/main gets one commit that changes
# utils.py, makes hermes_cli/config.py and run_agent.py import a new utils name, and adds a package. Runs the REAL autostash + _pull_updates from the install's own tree and the REAL entry point
# (`python -m hermes_cli.main config path`) under a disposable HOME/HERMES_HOME:
#   A  updater SIGKILLed before git wrote anything; user edits upstream-changed files, fetches, runs hermes
#   B  custom branch whose commit conflicts upstream: update exits 1; user merges by hand, runs hermes
#   C  git wrote config.py + the new package, SIGKILL; user edits an upstream-changed file git never wrote
#   D  C in a linked-worktree install (`.git` is a file)
#   E  custom branch whose commit merges cleanly: the kill lands inside the `git merge`, after it wrote
#      the merged run_agent.py; the user edits a file git never wrote, then runs `hermes-agent`'s import
# Fixed: A/B untouched, C/D/E restored with the user edit kept and the re-update lands; one VERDICT line.
# [python] defaults to <repo>/venv/bin/python (needs the Hermes deps).
set -u
REPO=$1; REF=$2; LABEL=$3
PY=${4:-$REPO/venv/bin/python}
U=$(mktemp -d -t "hermes-interrupted-pull-ab-$LABEL.XXXXXX")
export HOME=$U/fakehome HERMES_HOME=$U/fakehome/.hermes
unset PYTEST_CURRENT_TEST
cat > "$U/pull.py" <<'PYEOF'
# `hermes update` after its fetch: the REAL autostash + _pull_updates, imported from the install's tree.
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
import hermes_cli.main as m  # noqa: E402

assert pathlib.Path(m.__file__).resolve().is_relative_to(root.resolve()), m.__file__
m.PROJECT_ROOT = root
from hermes_cli import update_cmd  # noqa: E402

stash = m._stash_local_changes_if_needed(["git"], root)
try:
    update_cmd._pull_updates(["git"], "main", stash, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False)
    print("PULL: returned")
except SystemExit as e:
    print("PULL: SystemExit", e.code)
PYEOF

setup() {  # $1 = install kind: clone | worktree
  cd "$U"; rm -rf "$U/origin" "$U/install" "$U/wtbase"; mkdir -p "$HERMES_HOME"
  git clone -q --shared --no-checkout "$REPO" "$U/origin"
  ( cd "$U/origin" && git checkout -q -B main "$REF" && git config user.email t@x.invalid && git config user.name t
    printf '\n\ndef _torn_probe():\n    return 1\n' >> utils.py
    sed -i 's/^from utils import atomic_replace, fast_safe_load, file_signature$/from utils import atomic_replace, fast_safe_load, file_signature, _torn_probe  # noqa: F401/' hermes_cli/config.py
    grep -q _torn_probe hermes_cli/config.py || { echo "SETUP: config.py import line not found"; exit 1; }
    echo "from utils import _torn_probe  # noqa: E402,F401" >> run_agent.py
    mkdir -p torn_newpkg && echo "from utils import _torn_probe" > torn_newpkg/__init__.py
    git add -A && git commit -qm "upstream B" )
  if [ "$1" = worktree ]; then
    git clone -q --shared --no-checkout "$U/origin" "$U/wtbase"
    git -C "$U/wtbase" update-ref --no-deref HEAD origin/main  # frees `main` for the linked worktree
    git -C "$U/wtbase" worktree add -q -B main "$U/install" origin/main
    test -f "$U/install/.git" || { echo "SETUP: not a linked worktree"; exit 1; }
  else
    git clone -q --shared "$U/origin" "$U/install"
  fi
  cd "$U/install" && git config user.email t@x.invalid && git config user.name t
  git reset -q --hard HEAD~1
}
marker() { cat "$(git rev-parse --git-dir)/hermes-update-pull" 2>/dev/null | tr '\n' ' '; }
hermes() { PYTHONPATH=$U/install timeout 120 "$PY" -m hermes_cli.main config path 2>&1 | grep -v '^$' | head -6 | sed 's/^/    hermes> /'; }
pull_bg() {  # real _pull_updates in its own process group; fake git on PATH may stall/tear the ff
  setsid bash -c "PATH=$U/fakebin:\$PATH exec $PY $U/pull.py $U/install" > "$U/pull.log" 2>&1 &
  echo $!
}

mkdir -p "$U/fakebin"
echo "=============== $LABEL ($REF) ==============="

# ---- A: SIGKILL before git wrote anything; user re-applies edits, upstream moves on, user fetches -------
echo; echo "--- A: kill before git wrote anything, then user edits + fetch, then any hermes command"
setup clone >/dev/null
cat > "$U/fakebin/git" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "--ff-only" ] && sleep 30; done
exec /usr/bin/git "$@"
EOF
chmod +x "$U/fakebin/git"
PID=$(pull_bg); for i in $(seq 60); do [ -n "$(marker)" ] && break; sleep 0.5; done; sleep 1
kill -KILL -- -$PID; wait $PID 2>/dev/null
echo "  marker after kill: [$(marker)]"
echo "MY_UNCOMMITTED_WORK = 42" >> utils.py
( cd "$U/origin" && echo "# upstream C" >> cli.py && git commit -qam C )
git fetch -q origin
echo "# my cli tweak" >> cli.py
echo "  before hermes: [$(git status --porcelain | tr '\n' ' ')]"
hermes
echo "  after hermes:  [$(git status --porcelain | tr '\n' ' ')] marker=[$(marker)]"
A_OK=$([ "$(grep -c MY_UNCOMMITTED_WORK utils.py)$(grep -c 'my cli tweak' cli.py)" = 11 ] && echo 1 || echo 0)
echo "  RESULT A: utils.py edit kept=$(grep -c MY_UNCOMMITTED_WORK utils.py) cli.py edit kept=$(grep -c 'my cli tweak' cli.py)"

# ---- B: custom branch, merge conflict -> sys.exit(1) (not a kill); user merges by hand ---------------
echo; echo "--- B: update exits on a merge conflict; user follows the advice and resolves by hand"
setup clone >/dev/null
git checkout -q -b mywork && echo "# my local change" >> run_agent.py && git commit -qam "local work"
PATH=/usr/bin:$PATH $PY $U/pull.py $U/install 2>&1 | grep -E "Merge conflict|PULL" | sed 's/^/    pull> /'
echo "  marker after failed update: [$(marker)]"
git merge origin/main >/dev/null 2>&1
$PY - <<'EOF'
import re, pathlib
p = pathlib.Path("run_agent.py"); s = p.read_text()
s = re.sub(r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", r"\1\2", s, flags=re.S) + "# RESOLUTION WORK\n"
p.write_text(s)
EOF
echo "  mid-merge: MERGE_HEAD=$(test -f .git/MERGE_HEAD && echo yes || echo no) [$(git status --porcelain | tr '\n' ' ')]"
hermes
B_OK=$([ -f .git/MERGE_HEAD ] && [ "$(grep -c 'RESOLUTION WORK' run_agent.py)" = 1 ] && echo 1 || echo 0)
echo "  RESULT B: MERGE_HEAD=$(test -f .git/MERGE_HEAD && echo yes || echo no) [$(git status --porcelain | tr '\n' ' ')] resolution kept=$(grep -c 'RESOLUTION WORK' run_agent.py) marker=[$(marker)]"

# ---- C/D: a REAL torn tree (git wrote config.py + new pkg, then SIGKILL) + a user edit git never wrote --
CD_OK=0
for KIND in clone worktree; do
  echo; echo "--- $([ $KIND = clone ] && echo C || echo D): torn tree in a $KIND install, user edit on an upstream-changed file"
  setup $KIND >/dev/null
  cat > "$U/fakebin/git" <<'EOF'
#!/usr/bin/env bash
# The fast-forward writes two files of the new commit, holds index.lock, and is SIGKILLed.
for a in "$@"; do
  if [ "$a" = "--ff-only" ]; then
    /usr/bin/git show origin/main:hermes_cli/config.py > hermes_cli/config.py
    mkdir -p torn_newpkg && /usr/bin/git show origin/main:torn_newpkg/__init__.py > torn_newpkg/__init__.py
    touch "$(/usr/bin/git rev-parse --git-dir)/index.lock"
    sleep 30
  fi
done
exec /usr/bin/git "$@"
EOF
  chmod +x "$U/fakebin/git"
  PID=$(pull_bg); for i in $(seq 60); do [ -f torn_newpkg/__init__.py ] && break; sleep 0.5; done; sleep 1
  kill -KILL -- -$PID; wait $PID 2>/dev/null
  echo "  torn: HEAD=$(git rev-parse --short HEAD) [$(git status --porcelain | tr '\n' ' ')] marker=[$(marker | cut -c1-60)...]"
  sed -i "1a USER_EDIT = 'stash re-applied'" run_agent.py  # upstream changes this file too; git never wrote it
  hermes
  echo "  RESULT $([ $KIND = clone ] && echo C || echo D): [$(git status --porcelain | tr '\n' ' ')] user edit kept=$(grep -c "USER_EDIT" run_agent.py) torn_newpkg=$(test -e torn_newpkg && echo present || echo gone) marker=[$(marker)]"
  PATH=/usr/bin:$PATH $PY $U/pull.py $U/install 2>&1 | grep PULL | sed 's/^/    re-update> /'
  echo "  after re-update: HEAD==origin/main? $([ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && echo yes || echo no) user edit kept=$(grep -c USER_EDIT run_agent.py)"
  [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && [ "$(grep -c USER_EDIT run_agent.py)" = 1 ] && CD_OK=$((CD_OK + 1))
done
# ---- E: the custom-branch `git merge` wrote its merged run_agent.py, SIGKILL; `hermes-agent` launches ----
echo; echo "--- E: kill inside the custom-branch merge, then the hermes-agent entry point (run_agent)"
setup clone >/dev/null
git checkout -q -b mywork && sed -i "1a LOCAL_WORK = 1" run_agent.py && git commit -qam "local work that merges cleanly"
cat > "$U/fakebin/git" <<'EOF'
#!/usr/bin/env bash
# The merge writes run_agent.py as the merge of both sides (upstream's half imports a utils name git has
# not written yet), holds index.lock, and is SIGKILLed.
for a in "$@"; do
  if [ "$a" = "--no-edit" ]; then
    /usr/bin/git show "$(/usr/bin/git merge-tree --write-tree HEAD origin/main):run_agent.py" > run_agent.py
    touch "$(/usr/bin/git rev-parse --git-dir)/index.lock"
    sleep 30
  fi
done
exec /usr/bin/git "$@"
EOF
chmod +x "$U/fakebin/git"
PID=$(pull_bg); for i in $(seq 60); do grep -q _torn_probe run_agent.py && break; sleep 0.5; done; sleep 1
kill -KILL -- -$PID; wait $PID 2>/dev/null
echo "  torn: HEAD=$(git rev-parse --short HEAD) [$(git status --porcelain | tr '\n' ' ')] marker=[$(marker | cut -c1-60)...]"
sed -i '2a MY_UNCOMMITTED_WORK = 42' utils.py  # upstream changes this file too; git never wrote it
PYTHONPATH=$U/install timeout 120 "$PY" -c "import run_agent; print('run_agent imported')" 2>&1 | grep -v '^$' | tail -4 | sed 's/^/    hermes-agent> /'
echo "  RESULT E: [$(git status --porcelain | tr '\n' ' ')] user edit kept=$(grep -c MY_UNCOMMITTED_WORK utils.py) marker=[$(marker)]"
PATH=/usr/bin:$PATH $PY $U/pull.py $U/install 2>&1 | grep PULL | sed 's/^/    re-update> /'
E_MERGED=$(git merge-base --is-ancestor origin/main HEAD && grep -q LOCAL_WORK run_agent.py && echo yes || echo no)
echo "  after re-update: origin/main merged with the local commit? $E_MERGED user edit kept=$(grep -c MY_UNCOMMITTED_WORK utils.py)"
E_OK=$([ "$E_MERGED" = yes ] && [ "$(grep -c MY_UNCOMMITTED_WORK utils.py)" = 1 ] && echo 1 || echo 0)

echo
if [ "$A_OK$B_OK$CD_OK$E_OK" = 1121 ]; then echo "VERDICT: FIXED ($LABEL) — user work untouched in A/B, torn tree restored with user edits kept in C/D/E"
else echo "VERDICT: FIRES ($LABEL) — A_ok=$A_OK B_ok=$B_OK CD_ok=$CD_OK/2 E_ok=$E_OK"; fi
rm -rf "$U"
