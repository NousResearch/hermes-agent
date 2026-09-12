# Kit inventory

This directory is written by `oss-onboard` / `init-oss-repo.sh` from [verified-oss-loop](https://github.com/kvnloo/verified-oss-loop). It is not a second loop.

`inventory.yml` is the VCS-friendly provenance list:

- **kit** — copied from the standard. Re-running onboard updates it when you have not edited the file.
- **modified** — started as kit, then changed in this repo. Re-run skips it. `--force` takes the new kit file. Or mark it `local`.
- **local** — added here (or marked local). Never overwritten, including under `--force`.

New skills in a kit release appear on the next `oss-onboard` because they are missing dest files.

```bash
python3 .verified-oss-loop/kit-inventory.py show --root .
python3 .verified-oss-loop/kit-inventory.py mark --root . --path skills/YOUR/SKILL.md --source local
python3 .verified-oss-loop/rollout.py show
```

Rollout scheme (`rollout.yml`): Arch-style `rolling` by default. `staged` or `stable` for slower repos. See the kit's [docs/rollout.md](https://github.com/kvnloo/verified-oss-loop/blob/main/docs/rollout.md).
