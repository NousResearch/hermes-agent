# Setup notes

Use this sequence when helping a user set up the repo locally.

## 1. Clone

```bash
git clone https://github.com/darkzOGx/youtube-automation-agent.git
cd youtube-automation-agent
```

## 2. Inspect before running

```bash
python3 ~/.hermes/skills/media/youtube-automation-agent/scripts/youtube_automation_helper.py inspect --repo .
```

## 3. Install dependencies

```bash
npm install
```

## 4. Prepare local config files

```bash
cp .env.example .env
cp config/credentials.example.json config/credentials.json
```

The repo also expects YouTube OAuth tokens in:
- `config/tokens.json`

## 5. Run setup

```bash
npm run setup
```

## 6. Test

```bash
npm test
```

## 7. Start app

```bash
npm start
```

Expected local port from the repo docs and setup path:
- `3456`

## 8. Verify endpoints

```bash
python3 ~/.hermes/skills/media/youtube-automation-agent/scripts/youtube_automation_helper.py probe --base-url http://localhost:3456
```

Check these endpoints explicitly:
- `/health`
- `/schedule`
- `/analytics`

## What success looks like

- required repo files are present
- npm dependencies installed
- credentials exist
- local server starts
- `/health` returns a healthy payload
