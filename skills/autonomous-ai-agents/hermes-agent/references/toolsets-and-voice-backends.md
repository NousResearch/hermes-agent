# Toolset Enablement & Free Voice Backends

Use when a toolset "cannot be enabled", when a platform is missing a tool the CLI has, or when
turning on TTS/STT without a paid API key.

## A disabled toolset on a platform is an allowlist, not a broken install

`hermes tools list` prints **one platform** (cli by default). Before concluding a toolset is
unavailable, read the allowlist that actually governs it:

```bash
grep -n -A30 '^platform_toolsets:' ~/.hermes/config.yaml   # the ACTIVE per-platform allowlist
grep -n -A30 '^known_builtin_toolsets:' ~/.hermes/config.yaml  # bookkeeping: lists all known, incl. absent ones
```

`known_builtin_toolsets` looks authoritative and is not — it enumerates toolsets the platform
*could* have (it will list `stt`, `tts`, `discord`, `yuanbao` on a platform that has none). Only
`platform_toolsets` decides. Enable with the platform flag:

```bash
hermes tools enable --platform telegram tts stt video x_search cronjob
```

`hermes tools --summary` refuses to run without a TTY (`requires an interactive terminal`) —
verify from the config file and `hermes doctor`'s Tool Availability block instead.

## Do not enable a voice toolset whose provider has no credential

Check the provider before flipping the flag, or you ship a tool that errors on first use:

```bash
grep -n -A14 '^tts:' ~/.hermes/config.yaml   # tts.provider
grep -n -A8  '^stt:' ~/.hermes/config.yaml   # stt.provider
```

A common state is `provider: openai` with no `VOICE_TOOLS_OPENAI_KEY` anywhere in `~/.hermes/.env`.
List the env key *names* (never values) with `grep -oE '^[A-Za-z0-9_]+=' ~/.hermes/.env | tr -d '='` —
note that a greedy `^[A-Z_]*\(A\|B\)[A-Z_]*` pattern silently matches nothing; keep the regex simple.

## Free local backends, no API key

```bash
hermes tools post-setup piper            # TTS: local neural, ~14MB wheel, voices on first use
hermes tools post-setup faster_whisper   # STT: on-device Whisper, model ~150MB on first use
hermes config set tts.provider piper
hermes config set stt.provider local
```

`faster_whisper` is a valid post-setup key even though `hermes tools post-setup --help` omits it —
the hook keys in the help text are a subset; the authoritative list is the `post_setup=` argument
in `hermes_cli/tools_config.py`. Rows there also carry the provider name and whether it is free
(e.g. `Local Whisper  ★ recommended · free  stt_provider="local"  post_setup="faster_whisper"`).
Prefer a local backend over a paid one when the user has no matching key; xAI STT/TTS can reuse an
existing Grok OAuth login (`post_setup="xai_grok"`) when one is already authenticated.

`hermes config set stt.provider local` prints `'stt.provider' is not a recognized config key`.
That warning is benign: `provider` is a runtime key with no seeded default (unset = autodetect), so
it is saved and read normally. Do not "fix" it by renaming the key.

## Installing a backend stages a NEW runtime venv

The runtime venv is content-addressed by the dependency selection, so each post-setup install
creates another `~/.hermes/installs/<id>/environments/<hash>/venv` (several hundred MB each) and
prints `Restart Hermes to use them`. The old envs do not disappear on their own:

```bash
for d in ~/.hermes/installs/*/environments/*/venv; do
  n=$(basename $(dirname $d))
  p=$("$d/bin/python" -c 'import piper' 2>/dev/null && echo piper || echo -)
  w=$("$d/bin/python" -c 'import faster_whisper' 2>/dev/null && echo whisper || echo -)
  echo "$n : $p / $w"
done
```

**`hermes doctor`'s "Runtime venv … active in this process" is NOT the running gateway's env.** It
reports what a *freshly launched* process resolves, and that changes the moment you install a new
backend. After `post-setup piper` + `post-setup faster_whisper` the sequence was: gateway still
executing the old env, doctor already naming the new one, and the old one therefore looked stale
and deletable — deleting it takes the gateway down.

Find the real live env with `lsof`, not `ps` (the gateway's argv is a bare `python3 -I -c …` with no
venv path in it, so a `ps`-based check silently matches nothing and the guard never fires):

```bash
lsof -t ~/.hermes/state.db                      # the holders (gateway, Desktop app)
lsof -p <pid> | grep -o '/environments/[0-9a-f]*/venv'   # the env each one is running from
```

`~/.hermes/scripts/prune-stale-runtime-envs.py` implements this: it dry-runs by default, refuses to
proceed if it cannot identify the active env, and keeps the active env, any env held open by a
state.db holder, and the newest. Re-run with `--apply` to delete. Expect two passes: the pre-restart
run can only free the env that is neither active nor live; the rest frees after the restart.

A dead `environments/<hash>/venv/bin` entry in `~/Library/LaunchAgents/ai.hermes.gateway.plist`'s
`PATH` is cosmetic — the plist launches via `/usr/bin/osascript` and the gateway resolves its own
venv at runtime, so a hash there that no longer exists on disk is not a broken install.

A toolset change alone takes effect on `/reset`; a newly installed backend needs a **gateway
restart** to be importable at all. Recommend `/restart` (or `hermes gateway restart` from a shell
outside the gateway) when both changed — never restart from inside a gateway session.
