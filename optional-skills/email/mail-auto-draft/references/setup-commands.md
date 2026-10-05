# Setup Commands

Concrete command sequence for setting up a local Himalaya-based mail auto-reply workflow.

## 1. Verify tools

```bash
himalaya --version
python3 --version
systemctl --user --version
python3 -c "import yaml, requests"
```

## 2. Create project directories

```bash
PROJECT_DIR="$HOME/Projekte/Automation/mail-auto-draft"
mkdir -p "$PROJECT_DIR"/{drafts,logs,data,runtime,prompts,deploy/systemd}
```

## 3. Create local secrets file

```bash
mkdir -p "$HOME/.config/mail-auto-draft"
chmod 700 "$HOME/.config/mail-auto-draft"
cat > "$HOME/.config/mail-auto-draft/secrets.env" <<'EOF'
GMAIL_APP_PASSWORD='YOUR_APP_PASSWORD'
EOF
chmod 600 "$HOME/.config/mail-auto-draft/secrets.env"
```

## 4. Configure Himalaya for Gmail

Example auth commands inside `~/.config/himalaya/config.toml`:

```toml
backend.auth.cmd = "sh -lc '. $HOME/.config/mail-auto-draft/secrets.env && printf %s \"$GMAIL_APP_PASSWORD\"'"
message.send.backend.auth.cmd = "sh -lc '. $HOME/.config/mail-auto-draft/secrets.env && printf %s \"$GMAIL_APP_PASSWORD\"'"
```

## 5. Validate script syntax

```bash
cd "$PROJECT_DIR"
python3 -m py_compile process_inbox.py
```

## 6. Test in draft mode

```bash
cd "$PROJECT_DIR"
python3 process_inbox.py --mode draft --limit 5
```

Review:
- drafts in `drafts/`
- logs in `logs/mail_actions.jsonl`

## 7. Test in auto mode with one mail

```bash
cd "$PROJECT_DIR"
python3 process_inbox.py --mode auto --limit 1
```

## 8. Install systemd user units

```bash
PROJECT_DIR="$HOME/Projekte/Automation/mail-auto-draft"
mkdir -p "$HOME/.config/systemd/user"
sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$PROJECT_DIR/deploy/systemd/mail-auto-draft.service" > "$HOME/.config/systemd/user/mail-auto-draft.service"
cp "$PROJECT_DIR/deploy/systemd/mail-auto-draft.timer" "$HOME/.config/systemd/user/mail-auto-draft.timer"
systemctl --user daemon-reload
systemctl --user enable --now mail-auto-draft.timer
```

## 9. Start and inspect background processing

```bash
systemctl --user start mail-auto-draft.service
systemctl --user status mail-auto-draft.timer --no-pager
systemctl --user status mail-auto-draft.service --no-pager
journalctl --user -u mail-auto-draft.service -n 50 --no-pager
```

## 10. Day-to-day commands

Manual run:

```bash
cd "$PROJECT_DIR"
python3 process_inbox.py --limit 1
```

Stop timer:

```bash
systemctl --user stop mail-auto-draft.timer
```

Restart timer:

```bash
systemctl --user restart mail-auto-draft.timer
```
