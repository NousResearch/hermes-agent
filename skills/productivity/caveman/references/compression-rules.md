# Caveman Compression Rules — Quick Reference

## Remove (Always)
- Articles: a, an, the
- Filler: just, really, basically, actually, simply, essentially, generally
- Pleasantries: sure, certainly, of course, happy to, I'd be happy to
- Hedging: it might be worth, you could consider, it would be good to
- Redundant: "in order to" → "to", "make sure to" → "ensure", "the reason is because" → "because"
- Connective fluff: however, furthermore, additionally, in addition
- Tool-call narration: "Let me check...", "I'll now run..."
- Decorative tables/emoji (unless asked)

## Preserve EXACTLY (Never Modify)
- Code blocks (fenced ``` and indented)
- Inline code (`backtick content`)
- URLs and links (full URLs, markdown links)
- File paths (`/src/components/...`, `./config.yaml`)
- Commands (`npm install`, `git commit`, `docker build`)
- Technical terms (library names, API names, protocols, algorithms)
- Proper nouns (project names, people, companies)
- Dates, version numbers, numeric values
- Environment variables (`$HOME`, `NODE_ENV`)
- Standard tech acronyms: DB, API, HTTP, CI, CD, PR, LLM, GPU, CPU, RAM, SSD, JSON, YAML, TOML, SQL, HTML, CSS, JS, TS, PY, RS, GO, JAVA, CPP, C, SH, BASH, ZSH, FISH, CLI, GUI, TUI, UI, UX, SDK, IDE, LSP, MCP, ACP, SSH, TLS, SSL, JWT, OAuth, OIDC, SAML, LDAP, DNS, DHCP, TCP, UDP, IP, IPv4, IPv6, MAC, LAN, WAN, VPN, VLAN, NAT, ARP, ICMP, FTP, SFTP, SCP, Rsync, Cron, systemd, launchd, Docker, K8s, K3s, Helm, Istio, Prometheus, Grafana, Loki, Tempo, Jaeger, Zipkin, OpenTelemetry, OTel, SRE, DevOps, CI/CD, GitOps, ArgoCD, Flux, Terraform, Ansible, Chef, Puppet, Salt, Packer, Vagrant, Nomad, Consul, Vault, Boundary, Waypoint, Otto, Serf, Mesos, Marathon, Chronos, Singularity, Podman, Buildah, Skopeo, CRI-O, containerd, runc, Kata, gVisor, Firecracker, QEMU, KVM, Xen, Hyper-V, VMware, VirtualBox, Parallels, UTM, WSL, WSL2

## Compress (Apply Aggressively)
- Short synonyms: big/extensive, fix/solution, use/utilize, help/assist, get/obtain, show/display, find/locate, make/create, build/construct, check/verify, fix/resolve, add/implement, remove/delete, change/modify, update/upgrade, improve/enhance, optimize/performance
- Fragments OK: "Run tests before commit" > "You should always run tests before committing"
- Drop "you should", "make sure to", "remember to", "ensure that" — just state action
- Merge redundant bullets
- One example where multiple show same pattern

## Pattern
```
[thing] [action] [reason]. [next step].
```

## Auto-Clarity Boundaries
STOP caveman for:
- Security warnings
- Irreversible confirmations
- Ambiguous multi-step sequences
- User asks to clarify

## Intensity Examples

### Full (Default)
```
New object ref each render. Inline object prop = new ref = re-render. Wrap in `useMemo`.
```

### Ultra
```
Inline obj prop → new ref → re-render. `useMemo`.
```

### Wenyan-Full
```
每繪新生對象參照，故重繪；以 useMemo 包之則免。
```