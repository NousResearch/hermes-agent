<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
  <a href="README.id.md"><img src="https://img.shields.io/badge/Lang-Bahasa%20Indonesia-yellow?style=for-the-badge" alt="Bahasa Indonesia"></a>
</p>

**Agen AI dengan penyempurnaan diri yang dibangun oleh [Nous Research](https://nousresearch.com).** Ini adalah agen dengan loop pembelajaran bawaan — ia membuat skill dari pengalaman, memperbaikinya selama digunakan, mendorong dirinya untuk menyimpan pengetahuan, mencari percakapan masa lalunya sendiri, dan membangun model yang semakin mendalam tentang siapa Anda dari satu sesi ke sesi berikutnya. Jalankan di VPS $5, cluster GPU, atau infrastruktur serverless yang hampir tidak berbiaya saat idle. Hermes tidak terikat pada laptop Anda — berbicaralah dengannya melalui Telegram sementara ia bekerja di VM cloud.

Gunakan model apa pun yang Anda inginkan — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, endpoint Anda sendiri, dan [banyak lainnya](https://hermes-agent.nousresearch.com/docs/integrations/providers). Ganti model dengan `hermes model` — tanpa perubahan kode dan tanpa vendor lock-in.

<table>
<tr><td><b>Antarmuka terminal sungguhan</b></td><td>TUI lengkap dengan penyuntingan multi-baris, autocomplete slash-command, riwayat percakapan, interrupt-and-redirect, serta streaming output tool.</td></tr>
<tr><td><b>Hadir di tempat Anda berada</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, dan CLI — semuanya dari satu proses gateway. Transkripsi memo suara dan kesinambungan percakapan lintas platform.</td></tr>
<tr><td><b>Loop pembelajaran tertutup</b></td><td>Memori yang dikurasi agen dengan dorongan berkala. Pembuatan skill otomatis setelah tugas kompleks. Skill memperbaiki dirinya selama digunakan. Pencarian sesi FTS5 dengan peringkasan LLM untuk mengingat lintas sesi. Pemodelan pengguna dialektik <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Kompatibel dengan standar terbuka <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Otomatisasi terjadwal</b></td><td>Penjadwal cron bawaan dengan pengiriman ke platform mana pun. Laporan harian, backup malam hari, audit mingguan — semuanya dalam bahasa alami dan berjalan tanpa pengawasan.</td></tr>
<tr><td><b>Mendelegasikan dan menjalankan secara paralel</b></td><td>Jalankan subagen terisolasi untuk workstream paralel. Tulis skrip Python yang memanggil tool melalui RPC, sehingga pipeline multi-langkah dapat diringkas menjadi giliran tanpa biaya konteks tambahan.</td></tr>
<tr><td><b>Berjalan di mana saja, bukan hanya di laptop</b></td><td>Tujuh backend terminal — local, Docker, SSH, Singularity, Modal, Daytona, dan Vercel Sandbox. Daytona dan Modal menawarkan persistensi serverless — environment agen berhibernasi saat idle dan bangun sesuai permintaan, sehingga nyaris tanpa biaya antar-sesi. Jalankan di VPS $5 atau cluster GPU.</td></tr>
<tr><td><b>Siap untuk riset</b></td><td>Pembuatan trajectory secara batch dan kompresi trajectory untuk melatih generasi berikutnya dari model tool-calling.</td></tr>
</table>

---

## Instalasi Cepat

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (native, PowerShell)

> **Perhatian:** Windows native menjalankan Hermes tanpa WSL — CLI, gateway, TUI, dan tool semuanya bekerja secara native. Jika Anda lebih memilih WSL2, one-liner Linux/macOS di atas juga berfungsi di sana. Menemukan bug? Silakan [laporkan issue](https://github.com/NousResearch/hermes-agent/issues).

Jalankan ini di PowerShell:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Installer menangani semuanya: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **dan Git Bash portabel** (MinGit, diekstrak ke `%LOCALAPPDATA%\hermes\git` — tidak memerlukan admin dan sepenuhnya terisolasi dari instalasi Git sistem). Hermes menggunakan Git Bash bawaan ini untuk menjalankan perintah shell.

Jika Git sudah terpasang, installer akan mendeteksinya dan menggunakannya. Jika belum, unduhan MinGit sekitar 45MB sudah cukup — tidak akan menyentuh atau mengganggu Git sistem.

> **Android / Termux:** Jalur manual yang telah diuji didokumentasikan dalam [panduan Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux). Pada Termux, Hermes memasang extra `.[termux]` yang telah dikurasi karena extra penuh `.[all]` saat ini menarik dependensi voice yang tidak kompatibel dengan Android.
>
> **Windows:** Windows native didukung penuh — one-liner PowerShell di atas memasang semuanya. Jika Anda lebih memilih WSL2, perintah Linux juga berfungsi di sana. Instalasi Windows native berada di `%LOCALAPPDATA%\hermes`; WSL2 memasang di `~/.hermes` seperti Linux.

Setelah instalasi:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

### Pemecahan Masalah

#### Windows Defender atau antivirus menandai `uv.exe` sebagai malware

Jika antivirus Anda (Bitdefender, Windows Defender, dan sebagainya) mengarantina `uv.exe` dari folder `bin` Hermes (`%LOCALAPPDATA%\hermes\bin\uv.exe`), ini adalah **false positive**. File tersebut adalah `uv` milik Astral — package manager Python berbasis Rust yang dibundel Hermes untuk mengelola environment Python-nya. Mesin antivirus berbasis ML umum menandai binary Rust tanpa tanda tangan yang mengunduh dan memasang package.

**Untuk memverifikasi bahwa salinan Anda autentik:**

```powershell
# Install GitHub CLI if needed
winget install --id GitHub.cli

# Login to GitHub
gh auth login

# Run verification
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

Jika attestation menyatakan "Verification succeeded" dan baris terakhir mencetak `True`, salinan tersebut valid.

**Untuk memasukkan Hermes ke whitelist:**
- **Windows Defender:** Jalankan PowerShell sebagai Admin → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Tambahkan exception di konsol Bitdefender (Protection > Antivirus > Settings > Manage Exceptions)
- Masukkan **folder** ke whitelist, bukan hash file — Hermes memperbarui `uv` dan hash berubah pada setiap versi

Untuk konteks lebih lanjut, lihat laporan upstream Astral: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Memulai

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes config set   # Set individual config values
hermes config get   # Print individual config values
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard (configures everything at once)
hermes claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
hermes update       # Update to the latest version
hermes doctor       # Diagnose any issues
```

📖 **[Dokumentasi lengkap →](https://hermes-agent.nousresearch.com/docs/)**

---

## Lewati Pengumpulan API Key — Nous Portal

Hermes bekerja dengan penyedia apa pun yang Anda inginkan — itu tidak berubah. Namun jika Anda tidak ingin mengumpulkan lima API key terpisah untuk model, web search, image generation, TTS, dan cloud browser, **[Nous Portal](https://portal.nousresearch.com)** mencakup semuanya dalam satu subscription:

- **300+ model** — pilih salah satunya dengan `/model <name>`
- **Tool Gateway** — web search (Firecrawl), image generation (FAL), text-to-speech (OpenAI), cloud browser (Browser Use), semuanya dirutekan melalui subscription Anda. Tidak perlu akun tambahan.

Satu perintah dari instalasi baru:

```bash
hermes setup --portal
```

Perintah itu membuat Anda login melalui OAuth, menetapkan Nous sebagai provider, dan mengaktifkan Tool Gateway. Periksa apa yang sudah terhubung kapan saja dengan `hermes portal info`. Detail lengkap tersedia di [halaman dokumentasi Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Anda tetap dapat menggunakan API key sendiri per tool kapan saja — gateway ini bekerja per-backend, bukan all-or-nothing.

---

## Referensi Cepat CLI vs Perpesanan

Hermes memiliki dua entry point: mulai terminal UI dengan `hermes`, atau jalankan gateway dan berbicara dengannya dari Telegram, Discord, Slack, WhatsApp, Signal, atau Email. Setelah berada dalam percakapan, banyak slash command digunakan bersama pada kedua antarmuka.

| Aksi                           | CLI                                           | Platform perpesanan                                                               |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| Mulai mengobrol                | `hermes`                                      | Jalankan `hermes gateway setup` + `hermes gateway start`, lalu kirim pesan ke bot |
| Mulai percakapan baru          | `/new` atau `/reset`                          | `/new` atau `/reset`                                                               |
| Ganti model                    | `/model [provider:model]`                     | `/model [provider:model]`                                                        |
| Atur personality               | `/personality [name]`                         | `/personality [name]`                                                            |
| Ulangi atau batalkan giliran terakhir | `/retry`, `/undo`                      | `/retry`, `/undo`                                                                |
| Kompres konteks / cek penggunaan | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                    |
| Jelajahi skill                 | `/skills` atau `/<skill-name>`                | `/<skill-name>`                                                                  |
| Interupsi pekerjaan saat ini   | `Ctrl+C` atau kirim pesan baru                | `/stop` atau kirim pesan baru                                                    |
| Status khusus platform         | `/platforms`                                  | `/status`, `/sethome`                                                            |

Untuk daftar perintah lengkap, lihat [panduan CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) dan [panduan Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Dokumentasi

Semua dokumentasi tersedia di **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Bagian                                                                                              | Yang Dibahas                                                |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)                 | Instal → setup → percakapan pertama dalam 2 menit           |
| [Penggunaan CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                         | Perintah, keybinding, personality, sesi                     |
| [Konfigurasi](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                  | File konfigurasi, provider, model, semua opsi               |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)                | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant  |
| [Keamanan](https://hermes-agent.nousresearch.com/docs/user-guide/security)                          | Persetujuan perintah, pairing DM, isolasi container         |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)            | 40+ tool, sistem toolset, backend terminal                  |
| [Sistem Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)              | Memori prosedural, Skills Hub, membuat skill                |
| [Memori](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                     | Memori persisten, profil pengguna, praktik terbaik          |
| [Integrasi MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)                 | Hubungkan server MCP apa pun untuk kapabilitas tambahan     |
| [Penjadwalan Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)              | Tugas terjadwal dengan pengiriman ke platform               |
| [File Konteks](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)        | Konteks proyek yang membentuk setiap percakapan             |
| [Arsitektur](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)               | Struktur proyek, loop agen, class utama                     |
| [Berkontribusi](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)             | Setup pengembangan, proses PR, gaya kode                    |
| [Referensi CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                  | Semua perintah dan flag                                     |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Referensi lengkap environment variable                      |

---

## Migrasi dari OpenClaw

Jika Anda berpindah dari OpenClaw, Hermes dapat mengimpor pengaturan, memori, skill, dan API key secara otomatis.

**Saat setup pertama kali:** Wizard setup (`hermes setup`) otomatis mendeteksi `~/.openclaw` dan menawarkan migrasi sebelum konfigurasi dimulai.

**Kapan saja setelah instalasi:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

Yang akan diimpor:

- **SOUL.md** — file persona
- **Memories** — entri MEMORY.md dan USER.md
- **Skills** — skill buatan pengguna → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — pola persetujuan
- **Messaging settings** — konfigurasi platform, pengguna yang diizinkan, working directory
- **API keys** — secret dalam allowlist (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — file audio workspace
- **Workspace instructions** — AGENTS.md (dengan `--workspace-target`)

Lihat `hermes claw migrate --help` untuk semua opsi, atau gunakan skill `openclaw-migration` untuk migrasi interaktif yang dipandu agen dengan preview dry-run.

---

## Berkontribusi

Kami menyambut kontribusi! Lihat [Panduan Kontribusi](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) untuk setup pengembangan, gaya kode, dan proses PR.

Quick start untuk kontributor — gunakan installer standar, lalu bekerja dari
checkout git lengkap yang dibuat di `$HERMES_HOME/hermes-agent` (biasanya
`~/.hermes/hermes-agent`). Ini sesuai dengan layout yang digunakan oleh `hermes update`,
managed venv, lazy dependencies, gateway, dan tooling dokumentasi.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Fallback clone manual (untuk clone sementara/CI ketika Anda memang tidak ingin
menggunakan managed install layout):

Buat venv di luar source tree hasil clone — venv di dalam direktori
yang dioperasikan agen dapat terhapus oleh perintah relative-path yang dijalankan agen
terhadap checkout-nya sendiri, sehingga runtime yang sedang berjalan dapat rusak di tengah sesi.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Komunitas

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — server MCP kontrol desktop Linux untuk Hermes dan host MCP lainnya, dengan accessibility tree AT-SPI, input Wayland/X11, screenshot, dan penargetan window compositor.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — bridge WeChat komunitas: jalankan Hermes Agent dan OpenClaw pada akun WeChat yang sama.

---

## Lisensi

MIT — lihat [LICENSE](LICENSE).

Dibangun oleh [Nous Research](https://nousresearch.com).