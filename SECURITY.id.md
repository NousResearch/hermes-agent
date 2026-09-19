# Kebijakan Keamanan Hermes Agent

Dokumen ini menjelaskan model kepercayaan Hermes Agent, menetapkan satu-satunya
batas keamanan yang diperlakukan proyek sebagai komponen penopang utama, serta mendefinisikan
cakupan laporan kerentanan.

## 1. Melaporkan Kerentanan

Laporkan secara privat melalui [GitHub Security Advisories](https://github.com/NousResearch/hermes-agent/security/advisories/new)
atau **security@nousresearch.com**. Jangan membuka issue publik untuk
kerentanan keamanan. **Hermes Agent tidak menjalankan program bug
bounty.**

Laporan yang berguna mencakup:

- Deskripsi ringkas dan penilaian tingkat keparahan.
- Komponen yang terdampak, diidentifikasi dengan path file dan rentang baris
  (misalnya `path/to/file.py:120-145`).
- Detail environment (`hermes --version`, commit SHA, OS, versi Python).
- Reproduksi terhadap `main` atau rilis terbaru.
- Pernyataan mengenai batas kepercayaan pada §2 yang berhasil dilewati.

Harap baca §2 dan §3 sebelum mengirim laporan. Laporan yang hanya menunjukkan
keterbatasan heuristic in-process yang tidak diperlakukan kebijakan ini sebagai
batas keamanan akan ditutup sebagai di luar cakupan berdasarkan §3 — tetapi lihat §3.2:
laporan tersebut tetap diterima sebagai issue atau pull request biasa, hanya tidak
melalui kanal keamanan privat.

---

## 2. Model Kepercayaan

Hermes Agent adalah agen personal single-tenant. Postur keamanannya
berlapis, dan tidak semua lapisan memiliki bobot yang sama sebagai batas keamanan.
Pelapor dan operator sebaiknya menalar lapisan tersebut dengan istilah yang sama.

### 2.1 Definisi

- **Proses agen.** Interpreter Python yang menjalankan Hermes Agent,
  termasuk modul Python apa pun yang telah dimuat (skills, plugins,
  hook handlers).
- **Backend terminal.** Target eksekusi pluggable untuk tool
  `terminal()`. Default-nya menjalankan perintah langsung pada host.
  Backend lain menjalankan perintah di dalam container, cloud sandbox, atau
  remote host.
- **Permukaan input.** Kanal apa pun yang memasukkan konten ke dalam
  konteks agen: input operator, web fetches, email, pesan gateway,
  pembacaan file, respons server MCP, dan hasil tool.
- **Trust envelope.** Kumpulan resource yang secara implisit diberikan
  operator kepada Hermes Agent saat menjalankannya — biasanya apa pun yang
  dapat diakses oleh akun pengguna operator pada host.
- **Stance.** Pernyataan eksplisit dalam dokumentasi atau kode Hermes Agent
  mengenai bagaimana lapisan pengguna (adapter, UI, file writer,
  shell) harus memperlakukan output agen — misalnya "dashboard merender
  output agen sebagai HTML inert."

### 2.2 Batas: Isolasi Tingkat OS

**Satu-satunya batas keamanan terhadap LLM adversarial adalah sistem
operasi.** Tidak ada komponen di dalam proses agen yang merupakan
containment — bukan approval gate, bukan output redaction, bukan pattern
scanner, dan bukan tool allowlist. Setiap komponen in-process yang
menyaring output LLM adalah heuristic yang bekerja pada string yang
dapat dipengaruhi penyerang, dan kebijakan ini memperlakukannya demikian.

Hermes Agent mendukung dua postur isolasi tingkat OS. Keduanya menangani
ancaman yang berbeda dan operator harus memilihnya secara sadar.

#### Isolasi backend terminal

Backend terminal non-default menjalankan perintah shell yang dihasilkan LLM
di dalam container, remote host, atau cloud sandbox. Tool file
(`read_file`, `write_file`, `patch`) juga berjalan melalui backend ini,
karena diimplementasikan di atas kontrak shell — tool tersebut tidak dapat
menjangkau path yang tidak diekspos backend.

Yang dibatasi oleh postur ini: segala sesuatu yang dilakukan agen dengan
mengeluarkan operasi shell atau file. Yang **tidak** dibatasi: segala sesuatu
yang dilakukan agen di dalam proses Python-nya sendiri. Ini mencakup
tool code-execution (dijalankan sebagai subprocess host), subprocess MCP
(dijalankan dari environment agen), pemuatan plugin, dispatch hook, dan
pemuatan skill (semuanya diimpor ke interpreter agen).

Isolasi backend terminal adalah postur yang tepat ketika kekhawatirannya
adalah shell destruktif yang dihasilkan LLM atau penulisan file yang tidak
diinginkan, sementara operator lainnya dianggap tepercaya.

#### Pembungkusan seluruh proses

Pembungkusan seluruh proses menjalankan seluruh process tree agen di dalam
sandbox. Setiap code path — shell, code-execution, MCP, tool file,
plugin, hook, dan pemuatan skill — tunduk pada kebijakan filesystem,
network, process, dan, bila berlaku, inference yang sama.

Hermes Agent mendukung ini dengan dua cara:

- **Image Docker dan setup Compose milik Hermes Agent.** Lebih ringan;
  agen berjalan dalam container standar dengan mount dan kebijakan network
  yang dikonfigurasi operator.
- **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)**.
  OpenShell menyediakan sandbox per sesi dengan kebijakan deklaratif
  di lapisan filesystem, network (L7 egress), process/syscall, dan
  inference-routing. Kebijakan network dan inference dapat di-hot-reload.
  Credential diinjeksikan dari Provider store dan tidak pernah menyentuh
  filesystem sandbox.

Di bawah pembungkus seluruh proses, heuristic in-process Hermes Agent
(§2.4) berfungsi sebagai pencegahan kecelakaan yang dilapiskan di atas
batas keamanan nyata. Ini adalah postur yang didukung ketika agen menerima
konten dari permukaan yang tidak dikendalikan operator — web terbuka,
email masuk, channel multi-user, server MCP tidak tepercaya — serta untuk
deployment produksi atau bersama.

Operator yang menjalankan backend local default dengan permukaan input
tidak tepercaya, atau menjalankan sandbox backend terminal tetapi berharap
sandbox tersebut membatasi code path yang tidak melalui shell, beroperasi
di luar postur keamanan yang didukung.

### 2.3 Pembatasan Credential

Hermes Agent memfilter environment yang diteruskan ke komponen in-process
dengan tingkat kepercayaan lebih rendah: subprocess shell, subprocess MCP,
skrip cron job, dan child process code-execution. Credential seperti
provider API key dan gateway token dihapus secara default; variable yang
secara eksplisit dideklarasikan operator atau oleh skill yang dimuat akan
diteruskan.

Hal ini mengurangi eksfiltrasi tidak disengaja. Ini bukan containment.
Komponen apa pun yang berjalan di dalam proses agen (skills, plugins, hook
handlers) dapat membaca apa pun yang dapat dibaca agen itu sendiri,
termasuk credential dalam memori. Mitigasi terhadap komponen in-process
yang terkompromi adalah review operator sebelum instalasi (§2.4,
§2.5), bukan environment scrubbing.

### 2.4 Heuristic In-Process

Komponen berikut menyaring atau memperingatkan tentang perilaku LLM.
Komponen tersebut berguna. Komponen tersebut bukan batas keamanan.

- **Approval gate** mendeteksi pola shell destruktif yang umum dan
  meminta persetujuan operator sebelum eksekusi. Shell bersifat Turing-
  complete; denylist terhadap string shell secara struktural tidak
  lengkap. Gate menangkap kesalahan dalam mode kooperatif, bukan output
  adversarial.
- **Output redaction** menghapus pola yang menyerupai secret dari tampilan.
  Produsen output yang termotivasi akan dapat melewatinya.
- **Skills Guard** memindai konten skill yang dapat diinstal untuk pola
  injection. Ini adalah alat bantu review; batas untuk skill pihak ketiga
  adalah review operator sebelum instalasi. Mereview skill berarti
  membaca kode Python dan skripnya, bukan hanya deskripsi SKILL.md —
  skill dapat mengeksekusi Python arbitrer saat import.

### 2.5 Model Kepercayaan Plugin

Plugin dimuat ke dalam proses agen dan berjalan dengan privilege penuh agen:
plugin dapat membaca credential yang sama, memanggil tool yang sama,
mendaftarkan hook yang sama, dan mengimpor modul yang sama seperti komponen
apa pun yang dikirim di dalam tree. Batas untuk plugin pihak ketiga adalah
review operator sebelum instalasi — aturan yang sama seperti skill (§2.4),
dijelaskan terpisah karena plugin secara arsitektural lebih berat dan sering
membawa background service, network listener, serta dependensinya sendiri.

Plugin yang berbahaya atau buggy bukan merupakan kerentanan dalam Hermes Agent
itu sendiri. Bug pada jalur instalasi atau discovery plugin Hermes Agent yang
mencegah operator melihat apa yang mereka instal termasuk dalam cakupan §3.1.

### 2.6 Permukaan Eksternal

**Permukaan eksternal** adalah kanal apa pun di luar proses agen lokal yang
memungkinkan caller mengirim pekerjaan agen, menyelesaikan approval, atau
menerima output agen. Setiap permukaan memiliki model otorisasinya sendiri,
tetapi aturan berikut berlaku secara seragam.

**Permukaan dalam Hermes Agent:**

- **Adapter platform gateway.** Sebagian besar integrasi perpesanan dikirim
  sebagai plugin bawaan di `plugins/platforms/<name>/` (Telegram, Discord,
  Slack, email, SMS, dan sebagainya). Base type bersama dan sejumlah kecil
  adapter legacy/direct berada di `gateway/platforms/`
  (`base.py`, Signal, API server, webhooks, …), dengan discovery dan
  deferred loading melalui `gateway/platform_registry.py`.
- **Permukaan HTTP yang terekspos network.** Adapter API server, plugin
  dashboard, endpoint HTTP plugin kanban, serta plugin lain yang membuka
  listening socket.
- **Adapter editor / IDE.** Adapter ACP (`acp_adapter/`) dan integrasi
  sejenis yang menerima request dari proses client lokal.
- **Gateway TUI (`tui_gateway/`).** Backend JSON-RPC untuk Ink terminal UI,
  diakses melalui IPC lokal.

**Aturan seragam:**

1. **Otorisasi wajib pada setiap permukaan yang melintasi batas
   kepercayaan.** Untuk permukaan messaging dan HTTP network, batasnya adalah
   network: otorisasi berarti caller allowlist yang dikonfigurasi operator.
   Untuk permukaan editor dan local-IPC (ACP, gateway TUI), batasnya adalah
   akun pengguna host: otorisasi berarti mengandalkan access control tingkat
   OS (permission file, bind loopback-only) dan tidak mengekspos permukaan
   melampaui pengguna lokal tanpa lapisan auth network yang eksplisit.
2. **Allowlist wajib untuk setiap adapter network-exposed yang
   diaktifkan.** Adapter harus menolak mengirim pekerjaan agen, menyelesaikan
   approval, atau meneruskan output hingga allowlist ditetapkan. Code path
   yang fail-open ketika tidak ada allowlist dikonfigurasi merupakan bug kode
   yang termasuk cakupan §3.1.
3. **Identifier sesi adalah routing handle, bukan batas otorisasi.**
   Mengetahui session ID caller lain tidak memberikan akses ke approval atau
   output mereka; otorisasi selalu diperiksa ulang terhadap allowlist (atau
   ekuivalen tingkat OS).
4. **Di dalam set yang diotorisasi, semua caller sama-sama dipercaya.**
   Hermes Agent tidak memodelkan capability per-caller di dalam satu adapter.
   Operator yang membutuhkan pemisahan capability sebaiknya menjalankan
   instance agen terpisah dengan allowlist terpisah.
5. **Mengikat permukaan local-only ke interface non-loopback adalah keputusan
   break-glass operator (§3.2).** Dashboard dan server HTTP plugin lainnya
   default ke loopback; mengeksposnya melalui `--host 0.0.0.0` atau ekuivalen
   menjadikan hardening public-exposure (§4) tanggung jawab operator.

---

## 3. Cakupan

### 3.1 Termasuk Cakupan

- Escape dari postur isolasi tingkat OS yang dideklarasikan (§2.2): code path
  yang dikendalikan penyerang dapat menjangkau state yang diklaim postur
  tersebut sebagai terisolasi.
- Akses permukaan eksternal tanpa otorisasi: caller di luar set otorisasi
  yang dikonfigurasi (allowlist, atau ekuivalen tingkat OS untuk permukaan
  local-IPC) dapat mengirim pekerjaan, menerima output, atau menyelesaikan
  approval (§2.6).
- Eksfiltrasi credential: kebocoran credential operator atau material
  otorisasi sesi ke tujuan di luar trust envelope, melalui mekanisme yang
  seharusnya mencegahnya (bug environment scrubbing, logging adapter,
  transport error yang mengirim credential ke upstream, dan sebagainya).
- Pelanggaran dokumentasi model kepercayaan: kode berperilaku bertentangan
  dengan prediksi kebijakan ini, dokumentasi Hermes Agent sendiri, atau
  ekspektasi operator yang wajar — termasuk kasus ketika Hermes Agent telah
  mendokumentasikan stance tentang bagaimana output harus dirender oleh
  lapisan pengguna (dashboard, adapter gateway, file writer, shell) dan
  sebuah code path melanggar stance tersebut.

### 3.2 Di Luar Cakupan

"Di luar cakupan" di sini berarti "bukan kerentanan keamanan berdasarkan
kebijakan ini." Itu tidak berarti "tidak layak dilaporkan." Peningkatan
heuristic in-process, gagasan hardening, dan perbaikan UX tetap diterima
sebagai issue atau pull request biasa — approval gate selalu dapat menangkap
lebih banyak pola, redaction selalu dapat dibuat lebih baik, dan perilaku
adapter selalu dapat diperketat. Item-item ini hanya tidak melalui kanal
private-disclosure dan tidak menerima advisory.

- **Bypass heuristic in-process (§2.4)** — bypass regex approval-gate,
  bypass redaction, bypass pola Skills Guard, dan laporan serupa terhadap
  heuristic di masa depan. Komponen tersebut bukan batas keamanan;
  mengalahkannya bukan kerentanan berdasarkan kebijakan ini.
- **Prompt injection itu sendiri.** Membuat LLM menghasilkan output yang
  tidak biasa — melalui konten yang diinjeksi, hallucination, training
  artifact, atau sebab lain — bukan kerentanan dengan sendirinya. "Saya
  berhasil melakukan prompt injection" tanpa outcome §3.1 yang dirangkai
  bukan laporan yang dapat ditindaklanjuti berdasarkan kebijakan ini.
- **Konsekuensi dari postur isolasi yang dipilih.** Laporan bahwa code path
  yang beroperasi di dalam cakupan posturnya dapat melakukan hal yang memang
  diizinkan postur tersebut bukan kerentanan. Contoh: tool shell atau file
  menjangkau state host di bawah backend local; subprocess code-execution
  atau MCP menjangkau state host pada isolasi backend terminal yang hanya
  melakukan sandbox pada shell; laporan yang prasyaratnya memerlukan akses
  tulis yang telah ada terhadap file konfigurasi atau credential milik
  operator (resource tersebut sudah berada di dalam trust envelope).
- **Pengaturan break-glass yang terdokumentasi.** Trade-off yang dipilih
  operator dan secara eksplisit menonaktifkan perlindungan: `--insecure` dan
  flag sejenis pada dashboard atau komponen lain, approval yang dinonaktifkan,
  backend local di produksi, development profile yang melewati keamanan
  hermes-home, dan sejenisnya. Laporan terhadap konfigurasi tersebut bukan
  kerentanan — itulah fungsi flag tersebut.
- **Skill dan plugin kontribusi komunitas.** Skill pihak ketiga (termasuk
  repository skill komunitas) dan plugin pihak ketiga berada pada permukaan
  review operator, bukan permukaan kepercayaan Hermes Agent (§2.4, §2.5).
  Skill atau plugin yang melakukan tindakan berbahaya adalah failure mode
  yang diharapkan bila tidak direview, bukan kerentanan pada Hermes Agent.
  Bug pada jalur instalasi skill atau plugin Hermes Agent yang mencegah
  operator melihat apa yang sedang diinstal termasuk cakupan §3.1.
- **Eksposur publik tanpa kontrol eksternal.** Mengekspos gateway atau API ke
  internet publik tanpa autentikasi, VPN, atau firewall.
- **Pembatasan read/write tingkat tool pada postur yang mengizinkan shell.**
  Jika sebuah path dapat dijangkau melalui tool terminal, laporan bahwa tool
  file lain dapat menjangkaunya tidak menambah apa pun.

---

## 4. Hardening Deployment

Keputusan hardening terpenting adalah mencocokkan isolasi (§2.2) dengan
tingkat kepercayaan konten yang akan diterima agen. Selain itu:

- Jalankan agen sebagai pengguna non-root. Image container yang disediakan
  melakukan ini secara default.
- Simpan credential dalam file credential operator dengan permission ketat,
  jangan pernah di konfigurasi utama dan jangan pernah di version control.
  Di bawah OpenShell, gunakan Provider store alih-alih file credential pada disk.
- Jangan mengekspos gateway atau API ke internet publik tanpa perlindungan
  VPN, Tailscale, atau firewall. Di bawah OpenShell, gunakan lapisan kebijakan
  network untuk membatasi egress.
- Konfigurasikan caller allowlist untuk setiap adapter network-exposed yang
  Anda aktifkan (§2.6).
- Review skill dan plugin pihak ketiga sebelum instalasi (§2.4,
  §2.5). Untuk skill, ini berarti membaca Python dan skripnya,
  bukan hanya SKILL.md. Laporan Skills Guard dan install audit
  log adalah permukaan review.
- Hermes Agent menyertakan supply-chain guard untuk peluncuran server MCP
  dan untuk perubahan dependency / bundled-package di CI; lihat
  `CONTRIBUTING.id.md` untuk detail.

---

## 5. Pengungkapan

- **Jendela coordinated disclosure:** 90 hari sejak laporan, atau hingga
  perbaikan dirilis, mana yang lebih dahulu.
- **Kanal:** thread GHSA atau korespondensi email dengan
  security@nousresearch.com.
- **Kredit:** pelapor diberi kredit dalam release notes kecuali
  meminta anonimitas.