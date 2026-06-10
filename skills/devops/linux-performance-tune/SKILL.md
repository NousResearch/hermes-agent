---
name: linux-performance-tune
description: Diagnóstico e limpeza de performance em Linux Mint/Ubuntu para máquina de dev. RAM, swap, disco, processos zombies, caches.
trigger: PC lento, RAM cheia, swap saturado, disco cheio, load average alto, usuário reclama de desempenho.
---

# Linux Performance Tune

Diagnóstico e limpeza de performance em Linux Mint / Ubuntu com foco em máquina de dev.

## Diagnóstico (coletar dados)

```bash
# RAM + Swap
free -h && swapon --show

# Disco
df -h /

# Top CPU e Mem
ps aux --sort=-%cpu | head -15
ps aux --sort=-%mem | head -15

# Load + Uptime
uptime

# Temperatura
sensors 2>/dev/null | grep -E 'Tctl|edge'

# Detalhe memória
cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree'

# Services rodando
systemctl list-units --type=service --state=running --no-pager
```

## Limpeza padrão (executar na ordem)

### 1. Matar processos zombies / duplicados
- `next-server` rodando fora de terminal (normalmente root, usar `sudo kill -9 <PID>`)
- `chrome-devtools-mcp` órfãos E duplicados (mesmo processo 2x = ~500MB RAM desperdiçado)
- Qualquer processo com >10% RAM que não esteja sendo usado ativamente
- Verificar com `ps aux --sort=-%mem | head -15`
- **Dica:** `ps aux | grep chrome-devtools-mcp` — se retornar 2+ linhas com paths identicos, matar os de PID maior (sao os duplicados)

### 2. Reciclar Swap
```bash
sudo swapoff -a && sleep 2 && sudo swapon -a
# ZRAM: swapoff -a desativa o zram junto. swapon -a so reativa fstab entries (swapfile).
# Precisa reativar o zram manualmente via systemd:
sudo systemctl restart zram.service
```
Isso zera o swap. Só funciona se tiver RAM livre suficiente após matar processos.

**PITFALL**: `swapoff -a` desativa TODOS os swaps incluindo zram. `swapon -a` so reativa entradas do fstab (swapfile). O zram some silenciosamente — `swapon --show` mostra so o swapfile. Sempre rodar `sudo systemctl restart zram.service` depois para reativar o zram. Verificar com `zramctl && swapon --show`.

### 3. Limpar Docker
```bash
# Ver o que dá pra recuperar
docker system df

# Parar containers pesados antes do prune (SonarQube = ~1.85 GB RAM)
docker stop estacio-sonarqube 2>/dev/null

# Build cache — pode acumular 15GB+ silenciosamente
docker builder prune --all  # freed 16.76GB in one session

# Prune completo
docker system prune -af --volumes
```
Recupera MUITO espaço (containers parados, images, build cache).

**PITFALL**: `docker system prune -a` só remove images não usadas por NENHUM container (mesmo parado). Se quiser recuperar espaço de images de containers parados, pare e remova os containers primeiro: `docker rm <name>`, depois `docker image prune -a -f`

**PITFALL**: `docker builder prune --all` requer confirmação interativa (y/n). Pipe `echo "y"` or use `yes |` if scripting. The `--filter "until=24h"` flag with legacy builder may return 0B reclaimed — use `--all` instead.

### 4. Limpar node_modules órfãos
```bash
# Listar todos
find ~/ -maxdepth 3 -name "node_modules" -type d -prune 2>/dev/null | while read d; do echo "$d -> $(du -sh "$d" 2>/dev/null | cut -f1)"; done

# Remover projetos inativos (NÃO remover: forum ativo, hermes)
# Se der "Permissão negada", usar sudo
```

### 5. Limpar caches do sistema
```bash
sudo apt-get clean -y
sudo apt-get autoremove -y
sudo journalctl --vacuum-time=3d
rm -rf ~/.cache/thumbnails/*
rm -rf ~/.local/share/Trash/*
```

### 6. Drop page cache
```bash
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### 7. Desabilitar services inúteis (se usuário aprovar)
```bash
sudo systemctl disable --now ollama.service       # Se não usa LLM local
sudo systemctl disable --now avahi-daemon          # mDNS, raramente necessário
sudo systemctl disable --now cups cups-browsed     # Se não usa impressora
sudo systemctl disable --now ModemManager           # Se não usa modem
sudo systemctl disable --now anydesk                 # Se não usa acesso remoto
sudo systemctl disable --now waydroid-container     # Se não usa Android
sudo systemctl disable --now chrome-remote-desktop@<user>  # Se não usa

# Units estáticas (nao podem ser disabled — sem install config): colord, fwupd
# Se quiser parar: systemctl stop colord fwupd — mas voltam ao reboot
```

## Validação final
```bash
free -h
df -h /
uptime
sensors 2>/dev/null | grep Tctl
```

## Pitfalls
- next-server zombies podem ser root (verificar coluna USER no ps aux) → precisa sudo kill -9
- rocketseat-extractor-v2 node_modules tem arquivos com dono root → precisa sudo rm
- Docker prune pede confirmação → usar -y ou -af
- swapoff pode falhar se não houver RAM suficiente → matar processos PRIMEIRO, depois reciclar swap
- Não matar Hermes ativo (verificar pts do terminal atual)
- Load average demora ~5-15 min pra estabilizar após limpeza
- chrome-devtools-mcp duplicado consome ~500MB RAM — sempre verificar duplicatas, nao so orfaos. Padrao recorrente: 8-9 processos zumbis acumulados (~1 GB). Matar todos menos o mais recente: `pids=$(ps aux | grep '[c]hrome-devtools-mcp' | awk '{print $2}' | head -n -2); echo "$pids" | xargs kill`
- colord e fwupd sao units estáticas — `systemctl disable` retorna "Failed to disable unit: Unit file ... is not enabled" — usar `systemctl stop` apenas
- Reciclar swap ZERA o swap mas requer RAM livre suficiente — se swap tem 2GB e so sobraram 500MB RAM, NAO fazer swapoff

## Hardware do usuário
- AMD Ryzen 4600G (6c/12t) + Vega 6 integrada
- NVIDIA GT 710 dedicada (fraca, considera remover fisicamente)
- 16 GB RAM + 4 GB ZRAM (LZ4) + 2 GB swapfile = ~20 GB efetivo
- 120 GB NVMe (partição / com 120GB)
- Linux Mint 22.3 (Zena), kernel 6.17, XFCE 4.18

## Services que ficam rodando por padrão
- dockerd (~380 MB) — parar se não for usar: `sudo systemctl stop docker docker.socket containerd`
- SonarQube Docker — 3 processos Java (Elasticsearch ~860MB, WebServer ~490MB, CeServer ~310MB) = **~1.85 GB RAM**. Parar com `docker stop estacio-sonarqube` quando não estiver usando. Reativar com `docker start estacio-sonarqube`
- ollama — DESABILITADO por padrão (não usa)
- 4x Hermes (~1.3 GB total) — normal quando trabalhando
- Hindsight API (~1.8 GB) — essencial para memória entre sessões, não matar

## ZRAM Swap Comprimido (Opcional mas Recomendado)

ZRAM cria um block device comprimido na propria RAM. Com LZ4 (~2:1 ratio),
4 GB de zram consome ~2 GB de RAM real. Muito mais rapido que swap em disco.

### Setup Rapido

```bash
# Carregar modulo
sudo modprobe zram num_devices=1

# Configurar: 4GB comprimido, LZ4 (zramctl --streams NAO funciona em kernels novos)
sudo zramctl --find --size 4G --algorithm lz4

# Ativar como swap (prioridade alta = kernel usa ANTES do swapfile)
sudo mkswap /dev/zram0 > /dev/null 2>&1
sudo swapon -p 100 /dev/zram0

# Swappiness alta e ok com zram (e rapido, entao usar agressivamente)
sudo sysctl vm.swappiness=80
```

### Persistir Atraves de Reboots

Criar systemd service:

```bash
# Script de setup (no home do usuario, nao /root)
sudo bash -c 'cat > /opt/scripts/setup-zram.sh << "EOF"
#!/bin/bash
set -euo pipefail
sudo modprobe zram num_devices=1
[ -e /dev/zram0 ] || exit 1
swapon --show=NAME --noheadings | grep -q "/dev/zram0" && exit 0
sudo zramctl --reset /dev/zram0 2>/dev/null || true
sudo zramctl --find --size 4G --algorithm lz4
sudo mkswap /dev/zram0 > /dev/null 2>&1
sudo swapon -p 100 /dev/zram0
sudo sysctl vm.swappiness=80
EOF'
sudo chmod +x /opt/scripts/setup-zram.sh

# Systemd unit
sudo bash -c 'cat > /etc/systemd/system/zram.service << "EOF"
[Unit]
Description=Setup ZRAM compressed swap
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/scripts/setup-zram.sh
ExecStop=/bin/bash -c "swapoff /dev/zram0 && zramctl --reset /dev/zram0"

[Install]
WantedBy=multi-user.target
EOF'
sudo systemctl daemon-reload
sudo systemctl enable zram.service
```

### Persistir Swappiness

```bash
# Adicionar ao sysctl.conf
grep -q 'vm.swappiness' /etc/sysctl.conf && \
  sudo sed -i 's/vm.swappiness=.*/vm.swappiness=80/' /etc/sysctl.conf || \
  echo 'vm.swappiness=80' | sudo tee -a /etc/sysctl.conf
```

### Hibrido: ZRAM + Swapfile

Resultado final (6 GB total de swap):

```
NAME       TYPE      SIZE USED PRIO
/swapfile  file        2G  40M   -2    <- fallback (disco, lento)
/dev/zram0 partition   4G   0B  100    <- prioritario (RAM, rapido)
```

Kernel usa zram primeiro (prio 100). Swapfile so entra se zram encher.

### Verificacao

```bash
zramctl              # mostra algoritmo, tamanho, compressao
swapon --show        # mostra prioridades
free -h              # swap total = zram + swapfile
systemctl is-enabled zram.service  # deve ser "enabled"
```

### Capacidade Efetiva

Com 16 GB RAM + 4 GB zram (~2:1) + 2 GB swapfile:
- RAM real: 14 GB
- ZRAM logico: +4 GB (custa ~2 GB RAM real)
- Swap disco: +2 GB
- **Total: ~20 GB de memoria** (25% a mais)

### Pitfalls ZRAM

- **`--streams` flag**: NAO funciona em kernels 6.x+. Ignorar, usar `--find --size Ng --algorithm lz4` apenas.
- **zramctl tamanho**: O `--size` e o tamanho LOGICO (descomprimido). A RAM real consumida depende da compressao (~50% com LZ4 em uso normal).
- **Docker + ZRAM**: Docker reativa `ip_forward` ao subir containers. ZRAM funciona independente.
- **Swapoff zram**: `sudo swapoff /dev/zram0 && sudo zramctl --reset /dev/zram0` para desativar.
- **Reboot**: ZRAM e volatil por natureza. Service recria a cada boot (5 segundos).

## Referência de valores saudáveis
- RAM livre: > 2 GB
- Swap usado: < 500 MB
- Disco /: < 80%
- Load average: < 6 (para 12 threads)
- CPU temp (Tctl): < 75°C
