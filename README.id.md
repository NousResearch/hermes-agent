<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Dokumentasi"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/Lisensi-MIT-green?style=for-the-badge" alt="Lisensi: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Dibuat%20oleh-Nous%20Research-blueviolet?style=for-the-badge" alt="Dibuat oleh Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**Agen AI yang terus mengembangkan diri, dibuat oleh [Nous Research](https://nousresearch.com).** Ini satu-satunya agen dengan loop pembelajaran bawaan: ia menciptakan skill dari pengalaman, meningkatkannya saat digunakan, mendorong dirinya sendiri untuk menyimpan pengetahuan, mencari konversasi masa lalunya sendiri, dan membangun model yang semakin mendalam tentang siapa Anda lintas sesi. Jalankan di VPS seharga $5, kluster GPU, atau infrastruktur serverless yang nyaris tidak berbiaya saat idle. Ia tidak terikat pada laptop Anda: bicaralah dengannya dari Telegram sementara ia bekerja di VM cloud.

Gunakan model apa pun yang Anda mau: [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, endpoint Anda sendiri, dan [banyak lainnya](https://hermes-agent.nousresearch.com/docs/integrations/providers). Ganti dengan `hermes model`: tanpa perubahan kode, tanpa kunci vendor.

<table>
<tr><td><b>Antarmuka terminal yang sungguhan</b></td><td>TUI lengkap dengan penyuntingan multibaris, pelengkapan otomatis perintah garis-miring, riwayat percakapan, interupsi-dan-alihkan, serta keluaran alat streaming.</td></tr>
<tr><td><b>Hadir di tempat Anda berada</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, dan CLI: semua dari satu proses gateway. Transkripsi memo suara, kontinuitas percakapan lintas platform.</td></tr>
<tr><td><b>Loop pembelajaran tertutup</b></td><td>Memori yang dikurasi agen dengan pengingat berkala. Pembuatan skill otonom setelah tugas kompleks. Skill meningkat sendiri saat digunakan. Pencarian sesi FTS5 dengan rangkuman LLM untuk ingatan lintas sesi. Pemodelan pengguna dialektis <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Kompatibel dengan standar terbuka <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Automasi terjadwal</b></td><td>Penjadwal cron bawaan dengan pengiriman ke platform mana pun. Laporan harian, pencadangan malam hari, audit mingguan: semuanya dalam bahasa alami, berjalan tanpa pengawasan.</td></tr>
<tr><td><b>Mendelegasikan dan memparalelkan</b></td><td>Membuat subagen terisolasi untuk alur kerja paralel. Menulis skrip Python yang memanggil alat melalui RPC, menyederhanakan pipeline bertahap menjadi putaran berbiaya nol konteks.</td></tr>
<tr><td><b>Berjalan di mana saja, bukan hanya laptop Anda</b></td><td>Tujuh backend terminal: lokal, Docker, SSH, Singularity, Modal, Daytona, dan Vercel Sandbox. Daytona dan Modal menawarkan persistensi serverless: lingkungan agen Anda hibernasi saat idle dan bangun sesuai permintaan, nyaris tanpa biaya di antara sesi. Jalankan di VPS seharga $5 atau kluster GPU.</td></tr>
<tr><td><b>Siap untuk riset</b></td><td>Generasi trajektori secara massal, kompresi trajektori untuk melatih generasi model pemanggil-alat berikutnya.</td></tr>
</table>

---

## Instalasi Cepat

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (native, PowerShell)

> **Perhatian:** Windows native menjalankan Hermes tanpa WSL: CLI, gateway, TUI, dan alat semuanya bekerja secara native. Jika Anda lebih suka WSL2, perintah satu baris Linux/macOS di atas juga berfungsi di sana. Menemukan bug? Silakan [laporkan issue](https://github.com/NousResearch/hermes-agent/issues).

Jalankan ini di PowerShell:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Penginstal menangani semuanya: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **dan Git Bash portabel** (MinGit, dibongkar ke `%LOCALAPPDATA%\hermes\git`: tanpa izin admin, sepenuhnya terisolasi dari Git sistem mana pun). Hermes memakai Git Bash bawaan ini untuk menjalankan perintah shell.

Jika Git sudah terpasang, penginstal mendeteksinya dan memakainya sebagai gantinya. Jika tidak, unduhan MinGit ~45MB adalah semua yang Anda butuhkan: ia tidak akan menyentuh atau mengganggu Git sistem mana pun.

> **Android / Termux:** Jalur manual yang telah diuji didokumentasikan dalam [panduan Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux). Di Termux, Hermes memasang ekstra `.[termux]` yang dikurasi karena ekstra `.[all]` yang lengkap saat ini menarik dependensi suara yang tidak kompatibel dengan Android.
>
> **Windows:** Windows native didukung penuh: perintah satu baris PowerShell di atas menginstal semuanya. Jika Anda lebih suka WSL2, perintah Linux juga berfungsi di sana. Instalasi Windows native berada di bawah `%LOCALAPPDATA%\hermes`; instalasi WSL2 berada di bawah `~/.hermes` seperti di Linux.

Setelah instalasi:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

### Pemecahan Masalah

#### Windows Defender atau antivirus menandai `uv.exe` sebagai malware

Jika antivirus Anda (Bitdefender, Windows Defender, dll.) mengkarantina `uv.exe` dari folder `bin` Hermes (`%LOCALAPPDATA%\hermes\bin\uv.exe`), ini adalah **positif palsu**. File itu adalah `uv` milik Astral, manajer paket Python berbasis Rust yang disertakan Hermes untuk mengelola lingkungan Python-nya. Mesin antivirus berbasis ML umumnya menandai biner Rust yang tidak bertanda tangan yang mengunduh dan menginstal paket.

**Untuk memverifikasi bahwa salinan Anda asli:**

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

Jika attestation menyatakan "Verification succeeded" dan baris terakhir mencetak `True`, Anda aman.

**Untuk memasukkan Hermes ke daftar putih (whitelist):**
- **Windows Defender:** Jalankan PowerShell sebagai Admin → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Tambahkan pengecualian di konsol Bitdefender (Protection > Antivirus > Settings > Manage Exceptions)
- Masukkan ke daftar putih **folder**, bukan hash file: Hermes memperbarui `uv` dan hash berubah di setiap versi

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

## Lewati pengumpulan kunci API — Nous Portal

Hermes bekerja dengan provider apa pun yang Anda mau, itu tidak berubah. Tetapi jika Anda lebih suka tidak mengumpulkan lima kunci API terpisah untuk model, pencarian web, pembuatan gambar, TTS, dan browser cloud, **[Nous Portal](https://portal.nousresearch.com)** mencakup semuanya dalam satu langganan:

- **300+ model** — pilih salah satunya dengan `/model <name>`
- **Tool Gateway** — pencarian web (Firecrawl), pembuatan gambar (FAL), teks-ke-ucapan (OpenAI), browser cloud (Browser Use), semuanya dirutekan melalui langganan Anda. Tanpa akun tambahan.

Satu perintah dari instalasi baru:

```bash
hermes setup --portal
```

Itu masuk melalui OAuth, menetapkan Nous sebagai provider Anda, dan mengaktifkan Tool Gateway. Periksa apa saja yang sudah tersambung kapan pun dengan `hermes portal info`. Detail lengkap di [halaman docs Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Anda tetap dapat membawa kunci Anda sendiri per-alat kapan pun Anda mau: gateway bersifat per-backend, bukan semua-atau-tidak-ada.

---

## Referensi Cepat CLI vs. Pesan

Hermes memiliki dua titik masuk: mulai antarmuka terminal dengan `hermes`, atau jalankan gateway dan bicaralah dengannya dari Telegram, Discord, Slack, WhatsApp, Signal, atau Email. Setelah Anda dalam sebuah percakapan, banyak perintah garis-miring digunakan bersama di kedua antarmuka.

| Aksi                          | CLI                                           | Platform pesan                                                              |
| ----------------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| Mulai mengobrol               | `hermes`                                      | Jalankan `hermes gateway setup` + `hermes gateway start`, lalu kirim pesan ke bot |
| Mulai percakapan baru         | `/new` atau `/reset`                          | `/new` atau `/reset`                                                         |
| Ganti model                   | `/model [provider:model]`                     | `/model [provider:model]`                                                    |
| Tetapkan persona              | `/personality [name]`                         | `/personality [name]`                                                        |
| Ulangi atau batalkan putaran terakhir | `/retry`, `/undo`                     | `/retry`, `/undo`                                                            |
| Kompres konteks / cek penggunaan | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                     |
| Jelajahi skill                | `/skills` atau `/<skill-name>`                | `/<skill-name>`                                                              |
| Hentikan pekerjaan saat ini   | `Ctrl+C` atau kirim pesan baru                | `/stop` atau kirim pesan baru                                                |
| Status khusus platform        | `/platforms`                                  | `/status`, `/sethome`                                                        |

Untuk daftar perintah lengkap, lihat [panduan CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) dan [panduan Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Dokumentasi

Semua dokumentasi berada di **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Bagian                                                                                              | Yang Dicakup                                                |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [Memulai cepat](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)              | Instalasi → pengaturan → percakapan pertama dalam 2 menit   |
| [Penggunaan CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                         | Perintah, pintasan tombol, persona, sesi                     |
| [Konfigurasi](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                  | File konfigurasi, provider, model, semua opsi                |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)                | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant   |
| [Keamanan](https://hermes-agent.nousresearch.com/docs/user-guide/security)                          | Persetujuan perintah, pemasangan DM, isolasi kontainer       |
| [Alat & Toolset](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)              | 40+ alat, sistem toolset, backend terminal                   |
| [Sistem Skill](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)               | Memori prosedural, Skills Hub, membuat skill                 |
| [Memori](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                     | Memori persisten, profil pengguna, praktik terbaik           |
| [Integrasi MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)                 | Hubungkan server MCP mana pun untuk kemampuan yang diperluas |
| [Penjadwalan Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)             | Tugas terjadwal dengan pengiriman platform                   |
| [File Konteks](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)        | Konteks proyek yang membentuk setiap percakapan              |
| [Arsitektur](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)               | Struktur proyek, loop agen, kelas-kelas kunci                |
| [Berkontribusi](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)            | Pengaturan pengembangan, proses PR, gaya kode                |
| [Referensi CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                  | Semua perintah dan bendera                                  |
| [Variabel Lingkungan](https://hermes-agent.nousresearch.com/docs/reference/environment-variables)    | Referensi lengkap env var                                     |

---

## Migrasi dari OpenClaw

Jika Anda berasal dari OpenClaw, Hermes dapat mengimpor otomatis pengaturan, memori, skill, dan kunci API Anda.

**Selama pengaturan pertama kali:** Wizard pengaturan (`hermes setup`) otomatis mendeteksi `~/.openclaw` dan menawarkan migrasi sebelum konfigurasi dimulai.

**Kapan pun setelah instalasi:**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

Yang diimpor:

- **SOUL.md** — file persona
- **Memori** — entri MEMORY.md dan USER.md
- **Skill** — skill buatan pengguna → `~/.hermes/skills/openclaw-imports/`
- **Daftar izin perintah** — pola persetujuan
- **Pengaturan pesan** — konfigurasi platform, pengguna yang diizinkan, direktori kerja
- **Kunci API** — rahasia yang diizinkan (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **Aset TTS** — file audio workspace
- **Instruksi workspace** — AGENTS.md (dengan `--workspace-target`)

Lihat `hermes claw migrate --help` untuk semua opsi, atau gunakan skill `openclaw-migration` untuk migrasi interaktif yang dipandu agen dengan pratinjau dry-run.

---

## Berkontribusi

Kami menyambut kontribusi! Lihat [Panduan Kontribusi](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) untuk pengaturan pengembangan, gaya kode, dan proses PR.

Mulai cepat untuk kontributor: gunakan penginstal standar, lalu kerjakan dari
checkout git lengkap yang ia buat di `$HERMES_HOME/hermes-agent` (biasanya
`~/.hermes/hermes-agent`). Ini cocok dengan tata letak yang dipakai `hermes update`,
venv terkelola, dependensi malas, gateway, dan perkakas docs.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Fallback klon manual (untuk klon sekali pakai/CI tempat Anda sengaja tidak
menginginkan tata letak instalasi terkelola):

Buat venv di luar pohon sumber yang dikloning: venv di dalam direktori
tempat agen beroperasi dapat terhapus oleh perintah jalur-relatif yang dijalankan agen
terhadap checkout-nya sendiri, merusak runtime yang sedang berjalan di tengah sesi.

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
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux): server MCP kontrol desktop Linux untuk Hermes dan host MCP lain, dengan pohon aksesibilitas AT-SPI, input Wayland/X11, tangkapan layar, dan penargetan jendela kompositor.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw): jembatan WeChat komunitas: jalankan Hermes Agent dan OpenClaw di akun WeChat yang sama.

---

## Lisensi

MIT, lihat [LICENSE](LICENSE).

Dibuat oleh [Nous Research](https://nousresearch.com).