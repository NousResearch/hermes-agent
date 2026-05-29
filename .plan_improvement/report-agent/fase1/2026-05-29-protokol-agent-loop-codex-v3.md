## Summary

Protokol Agent-Loop Codex disempurnakan menjadi versi v3 yang lebih operasional, lebih jelas soal cadence 3-5 tahap, report file lengkap, GUI report ringkas, memory save, retry MCP, dan Definition of Done.

## Changes Made

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: menambahkan dokumen protokol baru yang siap dipakai sebagai arahan operasional.
  - OLD:
    ```md
    # file did not exist
    ```
  - NEW:
    ```md
    # Protokol Agent-Loop Codex v3

    > Status: rekomendasi operasional untuk Codex saat bekerja di repository ini.
    > Tujuan: menjaga pekerjaan coding tetap terverifikasi, terdokumentasi, dan
    > dikendalikan oleh leader project tanpa membuat laporan GUI terlalu sering.
    ```
  - Context: leader meminta menyempurnakan teks protokol Agent-Loop yang sebelumnya masih terlalu repetitif dan belum menjelaskan cadence tahap/report dengan presisi.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: memperjelas siklus inti.
  - OLD:
    ```text
    Retrieve -> Code -> Save -> GUI Report -> Respond
    ```
  - NEW:
    ```text
    Retrieve -> Code -> Save -> File Report -> GUI Report -> Respond
    ```
  - Context: leader menegaskan report file wajib dibuat, sehingga `File Report` perlu menjadi langkah eksplisit, bukan catatan di bawah GUI report.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: menambahkan aturan cadence laporan.
  - OLD:
    ```md
    - Panggil invoke_ui.py atau GUI Agent-Loop dengan laporan ringkas
    - Tunjukkan: apa yang dikerjakan, hasil verifikasi, status task
    ```
  - NEW:
    ```md
    Cadence laporan:

    - Untuk task besar, pecah pekerjaan menjadi sekitar 3-5 tahap besar.
    - Jangan melapor ke GUI untuk setiap micro-task kecil.
    - Buat report file lengkap untuk setiap tahap besar.
    - Laporan GUI cukup ringkas jika report file sudah lengkap.
    - Jika leader meminta detail di GUI, ikuti permintaan leader untuk turn itu.
    ```
  - Context: user/leader sebelumnya meminta laporan tidak terlalu sering dan GUI report dibuat ringkas jika file report sudah lengkap.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: memperjelas Retrieve.
  - OLD:
    ```md
    - Ambil memory project lokal dengan query semantik terkait task coding
    - Cek dokumentasi lokal project, architecture, conventions
    - Pahami context sebelum mulai coding
    ```
  - NEW:
    ```md
    Wajib dilakukan:

    1. Ambil memory project lokal dengan query semantik yang relevan.
    2. Baca instruksi lokal seperti `AGENTS.md`, dokumen arsitektur, conventions,
       dan file rencana yang disebut task.
    3. Verifikasi source code aktual sebelum mempercayai dokumen lama.
    4. Identifikasi file entry point, test terkait, dan risiko integrasi.
    5. Catat asumsi penting sebelum coding jika task besar atau berisiko.
    ```
  - Context: Retrieve perlu menghasilkan scope, file terkait, test plan, dan keputusan apakah task dibagi tahap.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: memperjelas Code dan standar implementasi.
  - OLD:
    ```md
    - Jalankan task coding: feature, bugfix, refactor
    - Gunakan path absolut untuk semua file
    - Lakukan verifikasi: tests, linting, build
    - Catat OLD vs NEW jika ada perubahan signifikan
    ```
  - NEW:
    ```md
    Standar implementasi:

    - Default ke opt-in untuk fitur baru.
    - Runtime behavior existing tidak boleh berubah tanpa config gate atau alasan eksplisit.
    - Observability tidak boleh memutus flow utama.
    - Error tracking/reporting harus fail-open kecuali task memang meminta hard-fail.
    - Jika ada integration point sensitif, buat foundation dan test dulu sebelum
      mengaktifkan behavior runtime yang lebih agresif.
    ```
  - Context: protocol v3 memasukkan prinsip teknis yang terbukti berguna selama implementasi provider gateway.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: memperjelas Save.
  - OLD:
    ```md
    - Jika milestone coding (fitur besar, refactor, setup), simpan ke project memory dengan event_type milestone
    - Jika bug fix, gunakan event_type bug_solved
    - Jika workflow/preferensi coding ditemukan, gunakan event_type user_preference
    ```
  - NEW:
    ```md
    Gunakan project memory bila ada milestone atau preferensi baru:

    - `event_type=milestone` untuk fitur besar, refactor, setup, atau tahap selesai.
    - `event_type=bug_solved` untuk bugfix dengan akar masalah dan bukti fix.
    - `event_type=user_preference` untuk preferensi workflow, format report, cadence,
      atau instruksi leader/user yang harus diingat.
    ```
  - Context: Save dibuat lebih eksplisit agar memory tidak terlalu banyak tetapi tetap menyimpan keputusan penting.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: menambahkan section File Report lengkap.
  - OLD:
    ```md
    - Tulis report: tulislah report di file baru juga
    ```
  - NEW:
    ```md
    ## 4. File Report

    Tujuan: meninggalkan artefak repo yang bisa dibaca ulang tanpa membuka transcript GUI.

    Lokasi:

    ```text
    .plan_improvement/report-agent/
    ```

    Nama file:

    ```text
    YYYY-MM-DD-<slug-tahap-atau-task>.md
    ```
    ```
  - Context: report file menjadi bagian first-class dari loop, lengkap dengan lokasi, nama file, dan 5 section wajib.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: memperjelas GUI Report dan retry MCP.
  - OLD:
    ```md
    - Jika ada kendala MCP → laporkan error dan coba lagi panggil mcp
    - Format: teks human-readable, bullets opsional, gunakan Indo/English mixed
    ```
  - NEW:
    ```md
    Jika GUI/MCP error:

    1. Coba lagi sampai maksimal 5 kali.
    2. Coba satu per satu, jangan paralel.
    3. Jika tetap gagal, tulis report file berisi error dan status blocked.
    4. Jangan klaim selesai tanpa laporan GUI berhasil atau tanpa instruksi eksplisit.
    ```
  - Context: retry rule dibuat tegas dan bounded supaya tidak menjadi loop tidak terkendali.

- **.plan_improvement/protokol-agent-loop-codex-v3.md**: menambahkan Definition of Done.
  - OLD:
    ```md
    # no explicit Definition of Done
    ```
  - NEW:
    ```md
    ## Definition of Done

    Sebuah tahap boleh disebut selesai jika:

    - Scope tahap sudah dikerjakan.
    - Test/lint/build relevan sudah dijalankan atau blocker dicatat.
    - Report file sudah dibuat.
    - Memory disimpan jika ada milestone/preferensi/bugfix.
    - GUI report sudah dikirim.
    - Leader memberi lanjut, selesai, atau arahan baru.
    ```
  - Context: DoD mencegah agen mengklaim selesai sebelum verifikasi/report/leader checkpoint terpenuhi.

## Technical Details

- Saya membuat dokumen baru, bukan mengubah `AGENTS.md`, karena teks yang diminta adalah protokol operasional lokal Agent-Loop dan lebih aman disimpan sebagai artifact `.plan_improvement/`.
- Struktur v3 memisahkan `File Report` dari `GUI Report`; ini menutup ambiguity versi sebelumnya.
- GUI report default dibuat ringkas karena report file sekarang wajib lengkap.
- Aturan cadence 3-5 tahap dimasukkan eksplisit agar task besar tidak menghasilkan spam GUI.
- Retry MCP dibatasi maksimal 5 kali, sesuai aturan leader sebelumnya, tetapi sekarang punya fallback status `blocked`.
- Path rule dibedakan:
  - Tools/code: path absolut.
  - Report: path relatif dari project root.
- Conflict priority ditambahkan: system/developer, repo instructions, leader terbaru, lalu dokumen rencana.
- Definition of Done ditambahkan untuk tahap dan batch.

## Results

- File protokol baru dibuat:
  - `.plan_improvement/protokol-agent-loop-codex-v3.md`
- Panjang dokumen:
  - 317 baris.
- Struktur utama terverifikasi dengan `rg`:
  - `# Protokol Agent-Loop Codex v3`
  - `## Prinsip Utama`
  - `## 1. Retrieve`
  - `## 2. Code`
  - `## 3. Save`
  - `## 4. File Report`
  - `## 5. GUI Report`
  - `## 6. Respond`
  - `## Aturan Keras`
  - `## Template Report File`
  - `## Template GUI Report Ringkas`
  - `## Definition of Done`

Validasi:

```text
rg -n "^## |^# " .plan_improvement/protokol-agent-loop-codex-v3.md
passed, headings found
```

```text
wc -l .plan_improvement/protokol-agent-loop-codex-v3.md
317 .plan_improvement/protokol-agent-loop-codex-v3.md
```

## What to Do Next / Things to Consider

- Jika leader ingin protokol ini menjadi aturan repository permanen, salin atau ringkas ke `AGENTS.md`.
- Jika hanya untuk Agent-Loop lokal, gunakan file v3 ini sebagai referensi utama dan jangan ubah upstream-facing docs.
- Jika protocol dipakai lint/evaluator, bisa dibuat checklist machine-readable di file terpisah.
- Pertimbangkan membuat versi pendek "Quick Protocol" 20-30 baris untuk ditempel ke prompt agent baru.
