[![Hermes Agent](https://github.com/NousResearch/hermes-agent/raw/main/assets/banner.png)](https://github.com/NousResearch/hermes-agent/blob/main/assets/banner.png)

# Hermes Agent ☤

[Hermes Agent](https://hermes-agent.nousresearch.com/) | [Hermes Desktop](https://hermes-agent.nousresearch.com/)

[![Documentation](https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge)](https://hermes-agent.nousresearch.com/docs/) [![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/NousResearch) [![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE) [![Built by Nous Research](https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge)](https://nousresearch.com) [![Türkçe](https://img.shields.io/badge/Lang-T%C3%BCrk%C3%A7e-red?style=for-the-badge)](https://github.com/NousResearch/hermes-agent/blob/main/README.tr.md)

**[Nous Research](https://nousresearch.com) tarafından geliştirilen, kendi kendini geliştiren yapay zeka ajanı.** Yerleşik bir öğrenme döngüsüne sahip tek ajan — deneyimlerden beceri üretir, kullanım sırasında bu becerileri geliştirir, kendine bilgiyi kalıcı hale getirmesi için hatırlatmalar yapar, kendi geçmiş konuşmalarında arama yapar ve oturumlar boyunca sizin kim olduğunuza dair giderek derinleşen bir model kurar. 5 dolarlık bir VPS'te, bir GPU kümesinde ya da boşta kaldığında neredeyse hiçbir maliyeti olmayan sunucusuz bir altyapıda çalıştırabilirsiniz. Dizüstü bilgisayarınıza bağımlı değildir — o bir bulut sanal makinede çalışırken siz Telegram'dan onunla konuşabilirsiniz.

İstediğiniz modeli kullanın — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, kendi endpoint'iniz ve [daha birçoğu](https://hermes-agent.nousresearch.com/docs/integrations/providers). `hermes model` ile değiştirin — kod değişikliği yok, kilitlenme yok.

| **Gerçek bir terminal arayüzü** | Çok satırlı düzenleme, slash-komut otomatik tamamlama, konuşma geçmişi, kesip-yönlendirme ve akan araç çıktısına sahip tam TUI. |
| --- | --- |
| **Nerede olursanız orada** | Telegram, Discord, Slack, WhatsApp, Signal ve CLI — hepsi tek bir gateway sürecinden. Sesli not deşifresi, platformlar arası konuşma sürekliliği. |
| **Kapalı bir öğrenme döngüsü** | Periyodik hatırlatmalarla ajan tarafından düzenlenen hafıza. Karmaşık görevlerden sonra otonom beceri oluşturma. Beceriler kullanım sırasında kendini geliştirir. Oturumlar arası hatırlama için LLM özetlemeli FTS5 oturum araması. [Honcho](https://github.com/plastic-labs/honcho) diyalektik kullanıcı modellemesi. [agentskills.io](https://agentskills.io) açık standardıyla uyumlu. |
| **Zamanlanmış otomasyonlar** | Herhangi bir platforma teslimatlı yerleşik cron zamanlayıcı. Günlük raporlar, gece yedeklemeleri, haftalık denetimler — hepsi doğal dilde, gözetimsiz çalışır. |
| **Devreder ve paralelleştirir** | Paralel iş akışları için izole alt-ajanlar oluşturur. Araçları RPC üzerinden çağıran Python betikleri yazar, çok adımlı boru hatlarını sıfır bağlam maliyetli turlara indirger. |
| **Sadece dizüstü bilgisayarınızda değil, her yerde çalışır** | Yedi terminal arka ucu — yerel, Docker, SSH, Singularity, Modal, Daytona ve Vercel Sandbox. Daytona ve Modal sunucusuz kalıcılık sunar — ajanınızın ortamı boştayken uyur, talep üzerine uyanır ve oturumlar arasında neredeyse hiçbir maliyeti olmaz. 5 dolarlık bir VPS'te ya da bir GPU kümesinde çalıştırın. |
| **Araştırmaya hazır** | Toplu yörünge (trajectory) üretimi, bir sonraki nesil araç-çağıran modelleri eğitmek için yörünge sıkıştırması. |

---

## Hızlı Kurulum

### Linux, macOS, WSL2, Termux

```
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (native, PowerShell)

> **Not:** Native Windows, Hermes'i WSL olmadan çalıştırır — CLI, gateway, TUI ve araçların hepsi yerel olarak çalışır. WSL2 kullanmayı tercih ediyorsanız yukarıdaki Linux/macOS tek satırlık komutu orada da çalışır. Bir hata mı buldunuz? Lütfen [issue açın](https://github.com/NousResearch/hermes-agent/issues).

PowerShell'de şunu çalıştırın:

```
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Kurulum programı her şeyi halleder: uv, Python 3.11, Node.js, ripgrep, ffmpeg ve **taşınabilir bir Git Bash** (MinGit, `%LOCALAPPDATA%\hermes\git` konumuna açılır — yönetici izni gerekmez, sistemdeki herhangi bir Git kurulumundan tamamen izole). Hermes, kabuk komutlarını çalıştırmak için bu paketlenmiş Git Bash'i kullanır.

Zaten Git kuruluysa, kurulum programı bunu algılar ve onu kullanır. Aksi takdirde yalnızca ~45MB'lık bir MinGit indirmesi yeterlidir — sistemdeki hiçbir Git kurulumuna dokunmaz veya onunla çakışmaz.

> **Android / Termux:** Test edilmiş manuel yol [Termux rehberinde](https://hermes-agent.nousresearch.com/docs/getting-started/termux) belgelenmiştir. Termux'ta Hermes, özel bir `.[termux]` eklentisi kurar çünkü tam `.[all]` eklentisi şu anda Android ile uyumsuz sesli bağımlılıkları içeriyor.
>
> **Windows:** Native Windows tamamen desteklenir — yukarıdaki PowerShell tek satırlık komutu her şeyi kurar. WSL2 kullanmayı tercih ediyorsanız Linux komutu orada da çalışır. Native Windows kurulumu `%LOCALAPPDATA%\hermes` altında; WSL2 kurulumu Linux'ta olduğu gibi `~/.hermes` altında yer alır.

Kurulumdan sonra:

```
source ~/.bashrc    # kabuğu yeniden yükle (veya: source ~/.zshrc)
hermes              # sohbete başlayın!
```

### Sorun Giderme

#### Windows Defender veya antivirüs `uv.exe` dosyasını zararlı yazılım olarak işaretliyor

Antivirüsünüz (Bitdefender, Windows Defender vb.) Hermes `bin` klasöründeki `uv.exe` dosyasını (`%LOCALAPPDATA%\hermes\bin\uv.exe`) karantinaya alıyorsa, bu bir **yanlış pozitif**tir. Bu dosya Astral'ın `uv` aracıdır — Hermes'in Python ortamını yönetmek için paketlediği Rust tabanlı Python paket yöneticisi. ML tabanlı antivirüs motorları, paket indirip kuran imzasız Rust ikili dosyalarını sıklıkla yanlışlıkla işaretler.

**Kopyanızın orijinal olduğunu doğrulamak için:**

```
# Gerekirse GitHub CLI'yi kurun
winget install --id GitHub.cli

# GitHub'a giriş yapın
gh auth login

# Doğrulamayı çalıştırın
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

"Verification succeeded" mesajı görürseniz ve son satır `True` yazdırıyorsa, sorun yok demektir.

**Hermes'i beyaz listeye eklemek için:**

- **Windows Defender:** PowerShell'i Yönetici olarak çalıştırın → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Bitdefender konsolunda bir istisna ekleyin (Koruma > Antivirüs > Ayarlar > İstisnaları Yönet)
- **Klasörü** beyaz listeye ekleyin, dosya hash'ini değil — Hermes, `uv`'yi günceller ve her sürümde hash değişir

Daha fazla bağlam için üst akış Astral raporlarına bakın: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Başlarken

```
hermes              # Etkileşimli CLI — bir konuşma başlatır
hermes model        # LLM sağlayıcınızı ve modelinizi seçin
hermes tools        # Hangi araçların etkin olduğunu yapılandırın
hermes config set   # Tekil yapılandırma değerlerini ayarlayın
hermes config get   # Tekil yapılandırma değerlerini yazdırın
hermes gateway      # Mesajlaşma gateway'ini başlatın (Telegram, Discord vb.)
hermes setup        # Tam kurulum sihirbazını çalıştırın (her şeyi tek seferde yapılandırır)
hermes claw migrate # OpenClaw'dan geçiş yapın (OpenClaw'dan geliyorsanız)
hermes update       # En son sürüme güncelleyin
hermes doctor       # Sorunları teşhis edin
```

📖 **[Tam dokümantasyon →](https://hermes-agent.nousresearch.com/docs/)**

---

## API anahtarı toplamayı atlayın — Nous Portal

Hermes istediğiniz sağlayıcıyla çalışır — bu değişmiyor. Ancak model, web araması, görsel üretimi, TTS ve bulut tarayıcısı için beş ayrı API anahtarı toplamak istemiyorsanız, **[Nous Portal](https://portal.nousresearch.com)** hepsini tek bir abonelik altında kapsar:

- **300+ model** — `/model <isim>` ile herhangi birini seçin
- **Tool Gateway** — web araması (Firecrawl), görsel üretimi (FAL), metinden sese (OpenAI), bulut tarayıcısı (Browser Use), hepsi aboneliğiniz üzerinden yönlendirilir. Ekstra hesap gerekmez.

Yeni bir kurulumdan tek komutla:

```
hermes setup --portal
```

Bu, OAuth ile giriş yapar, sağlayıcınızı Nous olarak ayarlar ve Tool Gateway'i açar. `hermes portal info` ile her an neyin bağlı olduğunu kontrol edin. Tüm detaylar için [Tool Gateway dokümantasyon sayfasına](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway) bakın.

Yine de istediğiniz zaman araç başına kendi anahtarlarınızı getirebilirsiniz — gateway, hepsi ya da hiçbiri değil, arka uç başınadır.

---

## CLI ve Mesajlaşma Hızlı Referans

Hermes'in iki giriş noktası vardır: `hermes` ile terminal arayüzünü başlatın, ya da gateway'i çalıştırıp Telegram, Discord, Slack, WhatsApp, Signal veya E-posta üzerinden konuşun. Bir konuşmaya girdikten sonra, birçok slash komutu her iki arayüzde de ortaktır.

| İşlem | CLI | Mesajlaşma platformları |
| --- | --- | --- |
| Sohbete başla | `hermes` | `hermes gateway setup` + `hermes gateway start` çalıştırın, ardından bota mesaj gönderin |
| Yeni konuşma başlat | `/new` veya `/reset` | `/new` veya `/reset` |
| Modeli değiştir | `/model [provider:model]` | `/model [provider:model]` |
| Bir kişilik ayarla | `/personality [isim]` | `/personality [isim]` |
| Son turu tekrarla veya geri al | `/retry`, `/undo` | `/retry`, `/undo` |
| Bağlamı sıkıştır / kullanımı kontrol et | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Becerilere göz at | `/skills` veya `/<beceri-adı>` | `/<beceri-adı>` |
| Mevcut işi kes | `Ctrl+C` veya yeni mesaj gönder | `/stop` veya yeni mesaj gönder |
| Platforma özel durum | `/platforms` | `/status`, `/sethome` |

Tam komut listeleri için [CLI rehberine](https://hermes-agent.nousresearch.com/docs/user-guide/cli) ve [Mesajlaşma Gateway rehberine](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) bakın.

---

## Dokümantasyon

Tüm dokümantasyon **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)** adresinde yer alır:

| Bölüm | Neyi Kapsar |
| --- | --- |
| [Hızlı Başlangıç](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | Kurulum → ayarlama → 2 dakikada ilk konuşma |
| [CLI Kullanımı](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | Komutlar, tuş bağlamaları, kişilikler, oturumlar |
| [Yapılandırma](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | Yapılandırma dosyası, sağlayıcılar, modeller, tüm seçenekler |
| [Mesajlaşma Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Güvenlik](https://hermes-agent.nousresearch.com/docs/user-guide/security) | Komut onayı, DM eşleştirme, konteyner izolasyonu |
| [Araçlar ve Araç Setleri](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ araç, araç seti sistemi, terminal arka uçları |
| [Beceri Sistemi](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Prosedürel hafıza, Skills Hub, beceri oluşturma |
| [Hafıza](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | Kalıcı hafıza, kullanıcı profilleri, en iyi uygulamalar |
| [MCP Entegrasyonu](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | Genişletilmiş yetenekler için herhangi bir MCP sunucusuna bağlanın |
| [Cron Zamanlama](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | Platforma teslimatlı zamanlanmış görevler |
| [Bağlam Dosyaları](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Her konuşmayı şekillendiren proje bağlamı |
| [Mimari](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | Proje yapısı, ajan döngüsü, temel sınıflar |
| [Katkıda Bulunma](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | Geliştirme kurulumu, PR süreci, kod stili |
| [CLI Referansı](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | Tüm komutlar ve bayraklar |
| [Ortam Değişkenleri](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Eksiksiz ortam değişkeni referansı |

---

## OpenClaw'dan Geçiş

OpenClaw'dan geliyorsanız, Hermes ayarlarınızı, hafızalarınızı, becerilerinizi ve API anahtarlarınızı otomatik olarak içe aktarabilir.

**İlk kurulum sırasında:** Kurulum sihirbazı (`hermes setup`) `~/.openclaw` dizinini otomatik olarak algılar ve yapılandırma başlamadan önce geçiş yapmayı önerir.

**Kurulumdan sonra herhangi bir zaman:**

```
hermes claw migrate              # Etkileşimli geçiş (tam ön ayar)
hermes claw migrate --dry-run    # Neyin taşınacağını önizle
hermes claw migrate --preset user-data   # Sırlar olmadan taşı
hermes claw migrate --overwrite  # Mevcut çakışmaların üzerine yaz
```

Neler içe aktarılır:

- **SOUL.md** — kişilik dosyası
- **Hafızalar** — MEMORY.md ve USER.md kayıtları
- **Beceriler** — kullanıcı tarafından oluşturulan beceriler → `~/.hermes/skills/openclaw-imports/`
- **Komut izin listesi** — onay kalıpları
- **Mesajlaşma ayarları** — platform yapılandırmaları, izin verilen kullanıcılar, çalışma dizini
- **API anahtarları** — izin listesine alınmış sırlar (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS varlıkları** — çalışma alanı ses dosyaları
- **Çalışma alanı talimatları** — AGENTS.md (`--workspace-target` ile)

Tüm seçenekler için `hermes claw migrate --help` komutuna bakın, ya da dry-run önizlemeleriyle etkileşimli, ajan rehberliğinde bir geçiş için `openclaw-migration` becerisini kullanın.

---

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Geliştirme kurulumu, kod stili ve PR süreci için [Katkı Rehberine](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) bakın.

Katkıda bulunanlar için hızlı başlangıç — standart kurulum programını kullanın, ardından onun oluşturduğu tam git checkout'undan çalışın: `$HERMES_HOME/hermes-agent` (genellikle `~/.hermes/hermes-agent`). Bu, `hermes update`'in, yönetilen venv'in, tembel bağımlılıkların, gateway'in ve dokümantasyon araçlarının kullandığı düzenle eşleşir.

```
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Manuel klonlama alternatifi (yönetilen kurulum düzenini kasıtlı olarak istemediğiniz tek kullanımlık klonlar/CI için):

Venv'i klonlanan kaynak ağacının dışında oluşturun — dizinin içindeki bir venv, ajanın kendi checkout'una karşı çalıştırdığı göreli yollu bir komut tarafından silinebilir ve bu da çalışan runtime'ı oturum ortasında yok edebilir.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Topluluk

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Hermes ve diğer MCP host'ları için AT-SPI erişilebilirlik ağaçları, Wayland/X11 girişi, ekran görüntüleri ve compositor pencere hedeflemesiyle Linux masaüstü kontrol MCP sunucusu.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Topluluk WeChat köprüsü: Aynı WeChat hesabında Hermes Agent ve OpenClaw'u çalıştırın.

---

## Lisans

MIT — bkz. [LICENSE](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE).

[Nous Research](https://nousresearch.com) tarafından geliştirildi.
