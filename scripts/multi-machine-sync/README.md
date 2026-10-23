# Hermes Shared Memory Multi-Machine Sync

This repository is for multi-machine Hermes Agent memory, configuration, and data synchronization.

## Architecture: One Master Multiple Slaves
- **Master**: Local authoritative server (maintains the canonical memory version)
- **Slaves**: Follow servers that pull sync from this repo
- **Sync Center**: This GitHub repo as neutral storage

## Synced Content
| Path in Repo  | Corresponding Path on Target Machine | Description |
|---------------|---------------------------------------|-------------|
| `hermes/memory/*` | `/opt/data/memories/*` | User and system persistent memory |
| `hermes-skills/*` | `/opt/data/hermes-skills/*` | Custom created/modified skills |
| `mempalace/*` | `/opt/data/mempalace/*` | Full MemPalace memory maze with vector index |
| `wikis/*` | `/opt/data/wikis/*` | Structured LLM Wiki knowledge base |

**Note**: `config.yaml` and `.env` are NOT synced. These are machine-specific deployment configurations and contain sensitive API keys. Each machine keeps its own local version, so you don't need to worry about overwriting or exposing secrets.

## NOT Synced (always kept local on each machine)
- Hermes source code
- Python virtual environments / installed packages
- Logs and cache files
- Session history and process state
- OS-specific binaries

## Sync Workflow

### Role Definitions
| Role | Script | Authority | Description |
|------|--------|-----------|-------------|
| **Master** | `./sync-master.sh` | 唯一权威，GitHub main必须和本机一致 | 基准机器，主数据维护者 |
| **Slave** | `./sync-slave.sh` (pull) / `./sync-slave-push.sh` (push) | 跟随主节点，可推送新增 | 其他机器，拉取基准+可贡献新增 |

---

### On Master (after updating memory/data):
```bash
cd ~/hermes-shared-memory
./sync-master.sh
```
- Automatically copies all content from this machine to repo
- Force pushes to GitHub main, guarantees main matches this authoritative machine
- Conflicts are automatically resolved in favor of this master machine

---

### On Slave:

#### 1. Pull latest authoritative sync from GitHub (daily use):
```bash
cd ~/hermes-shared-memory
./sync-slave.sh
```
- Pulls latest main from GitHub
- Copies all synced files to your local /opt/data/ paths
- Restart Hermes to load the new memory

#### 2. Push local new/modified memory to GitHub:
```bash
cd ~/hermes-shared-memory
./sync-slave-push.sh
```
- Pulls first, then copies local changes to repo
- Pushes to GitHub if no conflict
- **If conflict occurs**: automatically aborts, waits for master to resolve
- Master will merge and push authoritative version, you can then pull

---

### Conflict Resolution
- If slave pushes and causes conflict on GitHub: **Master resolves by running `./sync-master.sh`**
- Master always force pushes its own state to main, so master content always wins
- No confusing merge conflicts, master always has final say
## Last Sync
<!-- LAST_SYNC -->
Last synced: **2026-04-16** from Master
