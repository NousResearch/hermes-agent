# 🤖 CLAUDE - Project Memory & Critical Decisions

**Last Updated:** 2026-04-19  
**Project:** Republic AI Validator & GPU Miner Suite  
**Validator:** erhnNode (raivaloper1xnjkyyggaz54288vtrsle9wjwqz5tz4lttk8ka)

---

## 📊 CURRENT STATUS

### VPS Validator (154.12.118.199)
- **Status:** BOND_STATUS_BONDED ✅
- **Voting Power:** 10,789,275 RAI (10.7M)
- **Moniker:** erhnNode
- **Block:** 1,560,523+ (synced)
- **Uptime:** Active since Feb 26, 2026

### WSL GPU Miner (Local)
- **GPU:** NVIDIA RTX 4050 Laptop GPU
- **System:** Full-auto job processing
- **Tunnel:** Ngrok (auto-restart on boot)
- **Status:** ⚠️ Needs RAICOMPUTE setup

---

## 🎯 CRITICAL PROJECTS

### 1. republic-ai-node (Main Priority)
**Status:** In development (60% complete)

**Completed:**
- ✅ One-command installer script (`scripts/install.sh`)
- ✅ Health check script (`scripts/health_check.sh`)
- ✅ Moniker changed to erhnNode
- ✅ Basic documentation structure

**In Progress:**
- 🔄 Professional README (badges, screenshots, architecture)
- 🔄 Monitoring stack (Prometheus + Grafana)
- 🔄 CI/CD pipeline (GitHub Actions)

**Next Steps:**
1. Complete health_check.sh executable + test
2. Update README.md with badges and architecture diagram
3. Add monitoring dashboard
4. Create CI/CD pipeline
5. Write full documentation

### 2. republic-dashboard
**GitHub:** erhnysr/republic-dashboard  
**Purpose:** Real-time validator & GPU miner monitoring

**Features Needed:**
- Real-time WebSocket updates
- Historical charts (7/30 days)
- Dark/Light theme
- Export functionality
- Mobile responsive

### 3. republic-discord-bot
**Purpose:** Validator monitoring & alerts

**Features Needed:**
- Slash commands expansion
- Error handling
- Deployment guide

### 4. GPU Miner Dashboard
**Live:** https://republicstats.xyz  
**Bot:** t.me/republicgpuminerstatsbot  
**Status:** ✅ Production (1M RAI delegation received!)

**Features:**
- Live network stats
- Miner leaderboard (Top 25)
- Ecosystem directory
- Job tracking

---

## 🏆 MAJOR ACHIEVEMENTS

### 1M RAI Delegation (March 4, 2026)
**Reason:** Community contributions
- GitHub contributions & PRs
- Discord bot development
- GPU compute node setup
- Network improvements

**From:** 4,008 RAI → 1,008,340 RAI

### GPU Miner Dashboard
**Built:** republicstats.xyz + Telegram bot
- Scans chain from genesis
- Auto-updates every 500 blocks
- 24/7 monitoring
- Community praised

---

## 🔧 TECHNICAL DECISIONS

### Verification Issue Resolution
**Problem:** Sidecar verification returning False  
**Root Cause:** Volume mount missing in verification container  
**Solution:** Fixed verify.py to check multiple paths

**Updated verify.py:**
```python
paths_to_check = [
    '/output/result.bin',
    '/input/result.bin',
    '/app/result.bin',
    '/data/result.bin',
    '/results/result.bin',
    'result.bin'
]
```

**Result:** ✅ Verification now returns True

### Full-Auto vs Sidecar
**Decision:** Use Full-Auto (M4D's system)  
**Reason:**
- Sidecar has verification volume bug (unfixable without code change)
- Full-Auto proven working (40-50 jobs/hour)
- Better control and debugging
- Manual bech32 bug workaround included

### WSL Tunnel Management
**Problem:** Ngrok URL changes on WSL restart  
**Solution:** `start-tunnel.sh` script

**Features:**
- Auto-kills old ngrok
- Starts new tunnel
- Saves URL to `tunnel-url.txt`
- Restarts full-auto with new URL

**Usage:**
```bash
/home/erhn/start-tunnel.sh
```

---

## 📋 GPU MINER REWARD MODEL

**Weekly Pool:** 1,600,000 ecosystem points  
**Max/Wallet:** 240,000 points (15% cap)  
**Snapshot:** Every Saturday 23:59 UTC

### Scoring Formula:Final = Effort × JobCreation × Presence × Steadiness × Helpfulness × BuilderBonus

### Critical Requirements:
- **Minimum:** 5,000 completed jobs/week
- **Success Rate:** Keep high!
- **Presence:** 7/24 active hours
- **Steadiness:** Spread across all 7 days
- **Builder Bonus:** Validator uptime matters

### Reward Script:
```bash
git clone https://github.com/geofvn/Republic-gpuminer-rewardcal.git
cd Republic-gpuminer-rewardcal
python3 weekly_payout.py
```

---

## 🚨 CRITICAL ISSUES RESOLVED

### 1. Validator Unbonding + Jailed (April 14)
**Problem:** Validator went UNBONDING + jailed  
**Cause:** Node catching up, missed blocks  
**Solution:** Unjail command + wait for sync  
**Status:** ✅ Fixed (BOND_STATUS_BONDED)

### 2. Genesis File Corruption (Feb 26)
**Problem:** `error unmarshalling AppGenesis: invalid character '<'`  
**Solution:** `republicd comet unsafe-reset-all --keep-addr-book`  
**Result:** Network restarted, node recovered

### 3. Node Stuck at Block 372705 (Feb 18-26)
**Problem:** Node frozen for 8 days  
**Solution:** Network-wide restart  
**Lesson:** Explorer can cache, check node directly

---

## 📂 PROJECT STRUCTURE
republic-ai-node/
├── scripts/
│   ├── install.sh          ✅ One-command installer
│   ├── health_check.sh     ✅ Status checker
│   └── auto_compute.sh     ⚠️ Legacy
├── monitoring/
│   ├── dashboard.html
│   └── gpu_tracker.py
├── docs/
│   └── (empty - needs content)
├── README.md               🔄 Needs professional update
├── SETUP.md
├── TROUBLESHOOTING.md
├── MIGRATION.md
└── docker-compose.yml

---

## 🎯 IMMEDIATE PRIORITIES

### This Week:
1. ✅ Fix VPS validator (DONE)
2. 🔄 Setup RAICOMPUTE on WSL
3. 🔄 Complete republic-ai-node professional README
4. 🔄 Reach 5,000+ jobs for weekly rewards

### Next Week:
1. Monitoring stack (Prometheus + Grafana)
2. CI/CD pipeline
3. Full documentation
4. Video tutorial script

---

## 💡 LESSONS LEARNED

### 1. Always Verify State
- Explorer sites cache data
- Always check node directly: `republicd status`
- Validator status can change quickly

### 2. Unbonding Fix
- Unjail command alone not enough
- Need to wait for unbonding period OR
- Add large stake to bypass

### 3. Network Restarts
- Genesis corruption requires unsafe-reset-all
- Keep address book with --keep-addr-book
- Network can restart without warning

### 4. GPU Miner Success
- Verification must work for rewards
- Volume mounts critical for verification
- Test manually before production

### 5. Community Matters
- Contributions get recognized
- Discord bot = valuable tool
- GitHub PRs = delegation rewards

---

## 🔗 IMPORTANT LINKS

### VPS
- **IP:** 154.12.118.199
- **SSH:** `ssh root@154.12.118.199`
- **RPC:** tcp://154.12.118.199:26657
- **Cloudflare:** cup-massachusetts-absent-action.trycloudflare.com

### WSL
- **Home:** /home/erhn
- **Jobs:** /var/lib/republic/jobs
- **Logs:** ~/full-auto.log, ~/watchdog.log
- **Tunnel:** ~/tunnel-url.txt

### GitHub Repos
- **Main:** github.com/erhnysr/republic-ai-node
- **Dashboard:** github.com/erhnysr/republic-dashboard
- **Bot:** github.com/erhnysr/republic-discord-bot
- **Reference:** github.com/M4D2510/republic-ai-node

### Chain
- **Validator:** raivaloper1xnjkyyggaz54288vtrsle9wjwqz5tz4lttk8ka
- **Wallet:** rai1xnjkyyggaz54288vtrsle9wjwqz5tz4lv7k6at
- **Explorer:** explorer.rpcdot.com/republic-testnet

### Live Services
- **Dashboard:** republicstats.xyz
- **Bot:** t.me/republicgpuminerstatsbot
- **GitHub:** github.com/0xDarkSeidBull/republic-miner-dashboard

---

## 🔐 SECURITY NOTES

### Keys & Passwords
- **VPS Keyring:** File backend
- **WSL Keyring:** Test backend
- **Private Key:** Stored securely (not in docs)

### SSH Access
- **VPS:** Password auth
- **WSL:** Local only

---

## 📝 COMMANDS CHEAT SHEET

### VPS Health Check
```bash
echo "=== REPUBLIC VALIDATOR DURUMU ===" && \
systemctl status republicd --no-pager | head -3 && \
republicd status 2>&1 | grep -E "latest_block_height|catching_up|voting_power" && \
republicd query staking validator raivaloper1xnjkyyggaz54288vtrsle9wjwqz5tz4lttk8ka --home /root/.republic | grep -E "status:|tokens:"
```

### WSL Full Status
```bash
echo "=== NODE ===" && republicd status --node tcp://154.12.118.199:26657 2>/dev/null | jq '{block: .sync_info.latest_block_height, catching_up: .sync_info.catching_up}' && \
echo "=== FULL-AUTO ===" && ps aux | grep full-auto | grep -v grep && \
echo "=== GPU ===" && nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv,noheader && \
echo "=== JOBS ===" && ls /var/lib/republic/jobs | wc -l
```

### Start Tunnel + Full-Auto
```bash
/home/erhn/start-tunnel.sh
```

### Unjail Validator
```bash
republicd tx slashing unjail \
  --from validator \
  --chain-id raitestnet_77701-1 \
  --gas auto \
  --gas-adjustment 1.5 \
  --gas-prices 1000000000arai \
  --node tcp://localhost:26657 \
  --home /root/.republic \
  --keyring-backend file \
  -y
```

---

## 🎬 NEXT SESSION CHECKLIST

When Claude starts a new session:

1. Read this file first
2. Check VPS validator status
3. Check WSL GPU miner status
4. Review current priorities
5. Continue from "IMMEDIATE PRIORITIES"

---

**Remember:** This project earned 1M RAI delegation through community contributions. Keep building, keep contributing! 🚀
