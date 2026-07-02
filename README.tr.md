<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/D%C3%B6k%C3%BCmantasyon-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Dökümantasyon"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/Lisans-MIT-green?style=for-the-badge" alt="Lisans: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Nous%20Research%20Taraf%C4%B1ndan%20Geli%C5%9Ftirildi-blueviolet?style=for-the-badge" alt="Nous Research Tarafından Geliştirildi"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Espa%C3%B1ol-orange?style=for-the-badge" alt="Español"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.tr.md"><img src="https://img.shields.io/badge/Lang-T%C3%BCrk%C3%A7e-turquoise?style=for-the-badge" alt="Türkçe"></a>
</p>

**[Nous Research](https://nousresearch.com) tarafından geliştirilen kendini geliştiren AI ajanı.** Dahili öğrenme döngüsüne sahip tek ajandır — deneyimlerden beceriler oluşturur, kullanım sırasında onları iyileştirir, bilgiyi kalıcı kılmak için kendini dürtükler, geçmiş konuşmalarında arama yapar ve oturumlar boyunca kim olduğunuza dair giderek derinleşen bir model oluşturur. 5 dolarlık bir VPS'te, bir GPU kümesinde veya boştayken neredeyse hiçbir maliyeti olmayan sunucusuz altyapıda çalıştırın. Dizüstü bilgisayarınıza bağlı değildir — buluttaki bir VM'de çalışırken onunla Telegram'dan konuşun.

İstediğiniz modeli kullanın — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (200+ model), [NovitaAI](https://novita.ai) (Model API, Agent Sandbox ve GPU Cloud için AI-native bulut), [NVIDIA NIM](https://build.nvidia.com) (Nemotron), [Xiaomi MiMo](https://platform.xiaomimimo.com), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), [Hugging Face](https://huggingface.co), OpenAI veya kendi endpoint'iniz. `hermes model` ile değiştirin — kod değişikliği yok, kilitlenme yok.

<table>
<tr><td><b>Gerçek bir terminal arayüzü</b></td><td>Çok satırlı düzenleme, slash-komut otomatik tamamlama, konuşma geçmişi, kesinti ve yönlendirme ve akışkan araç çıktısına sahip tam TUI.</td></tr>
<tr><td><b>Nerede yaşarsanız orada çalışır</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal ve CLI — hepsi tek bir gateway sürecinden. Sesli not dökümü, platformlar arası konuşma sürekliliği.</td></tr>
<tr><td><b>Kapalı bir öğrenme döngüsü</b></td><td>Ajan tarafından düzenlenmiş bellek, periyodik dürtüklemelerle. Karmaşık görevlerden sonra otonom beceri oluşturma. Beceriler kullanım sırasında kendini geliştirir. Oturumlar arası hatırlama için LLM özetlemeli FTS5 oturum araması. <a href="https://github.com/plastic-labs/honcho">Honcho</a> diyalektik kullanıcı modelleme. <a href="https://agentskills.io">agentskills.io</a> açık standardı ile uyumlu.</td></tr>
<tr><td><b>Zamanlanmış otomasyonlar</b></td><td>Herhangi bir platforma teslimat ile yerleşik cron zamanlayıcı. Günlük raporlar, gece yedeklemeleri, haftalık denetimler — tamamen doğal dilde, gözetimsiz çalışır.</td></tr>
<tr><td><b>Delege eder ve paralelleştirir</b></td><td>Paralel iş akışları için izole alt ajanlar oluşturun. Araçları RPC aracılığıyla çağıran Python betikleri yazın, çok adımlı pipeline'ları sıfır bağlam maliyetli turlara dönüştürün.</td></tr>
<tr><td><b>Her yerde çalışır, sadece dizüstü bilgisayarınızda değil</b></td><td>Altı terminal arka ucu — yerel, Docker, SSH, Singularity, Modal ve Daytona. Daytona ve Modal sunucusuz kalıcılık sunar — ajan ortamınız boştayken hazırda bekler ve ihtiyaç duyulduğunda uyanır, oturumlar arasında neredeyse hiçbir maliyeti olmaz. 5 dolarlık bir VPS'te veya bir GPU kümesinde çalıştırın.</td></tr>
<tr><td><b>Araştırmaya hazır</b></td><td>Toplu trajectory oluşturma, bir sonraki nesil araç çağıran modelleri eğitmek için trajectory sıkıştırma.</td></tr>
</table>

---

## Hızlı Kurulum

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (yerel, PowerShell)

> **Uyarı:** Yerel Windows, Hermes'i WSL olmadan çalıştırır — CLI, gateway, TUI ve araçların hepsi yerel olarak çalışır. WSL2 kullanmayı tercih ederseniz, yukarıdaki Linux/macOS tek satırlık komut orada da çalışır. Bir hata mı buldunuz? Lütfen [issue açın](https://github.com/NousResearch/hermes-agent/issues).

Bunu PowerShell'de çalıştırın:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Yükleyici her şeyi halleder: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **ve taşınabilir bir Git Bash** (MinGit, `%LOCALAPPDATA%\hermes\git` konumuna çıkarılır — yönetici gerektirmez, sistemdeki herhangi bir Git kurulumundan tamamen izole edilmiştir). Hermes, shell komutlarını çalıştırmak için bu paketlenmiş Git Bash'i kullanır.

Zaten Git yüklüyse, yükleyici bunu algılar ve onun yerine kullanır. Aksi takdirde, ~45 MB'lık bir MinGit indirmesi yeterlidir — sisteminizdeki hiçbir Git'e dokunmaz veya müdahale etmez.

> **Android / Termux:** Test edilmiş manuel yol, [Termux rehberinde](https://hermes-agent.nousresearch.com/docs/getting-started/termux) belgelenmiştir. Termux'ta Hermes, düzenlenmiş bir `.[termux]` eklentisi kurar çünkü tam `.[all]` eklentisi şu anda Android ile uyumsuz ses bağımlılıklarını çeker.
>
> **Windows:** Yerel Windows tamamen desteklenir — yukarıdaki PowerShell tek satırlık komutu her şeyi kurar. WSL2 kullanmayı tercih ederseniz, Linux komutu da orada çalışır. Yerel Windows kurulumu `%LOCALAPPDATA%\hermes` altında yaşar; WSL2, Linux'taki gibi `~/.hermes` altına kurar.

Kurulumdan sonra:

```bash
source ~/.bashrc    # shell'i yeniden yükle (veya: source ~/.zshrc)
hermes              # sohbete başla!
```

### Sorun Giderme

#### Windows Defender veya antivirüs `uv.exe` dosyasını kötü amaçlı yazılım olarak işaretliyor

Antivirüsünüz (Bitdefender, Windows Defender vb.) Hermes `bin` klasöründeki (`%LOCALAPPDATA%\hermes\bin\uv.exe`) `uv.exe` dosyasını karantinaya alırsa, bu bir **yanlış pozitif**tir. Bu dosya, Hermes'in Python ortamını yönetmek için paketlediği Rust tabanlı Python paket yöneticisi olan Astral'in `uv` aracıdır. ML tabanlı antivirüs motorları, imzasız Rust ikili dosyalarını sıklıkla kötü amaçlı olarak işaretler.

**Kopyanızın orijinal olduğunu doğrulamak için:**

```powershell
# Gerekirse GitHub CLI'yi yükleyin
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

Doğrulama "Verification succeeded" diyorsa ve son satır `True` yazdırıyorsa, sorun yok.

**Hermes'i beyaz listeye eklemek için:**
- **Windows Defender:** PowerShell'i Yönetici olarak çalıştırın → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Bitdefender konsolunda bir istisna ekleyin (Koruma > Antivirüs > Ayarlar > İstisnaları Yönet)
- **Klasörü** beyaz listeye ekleyin, dosya hash'ini değil — Hermes `uv`'yi günceller ve hash her sürümde değişir

Daha fazla bağlam için yukarı akış Astral raporlarına bakın: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Başlarken

```bash
hermes              # Etkileşimli CLI — bir konuşma başlatın
hermes model        # LLM sağlayıcınızı ve modelinizi seçin
hermes tools        # Hangi araçların etkin olduğunu yapılandırın
hermes config set   # Bireysel yapılandırma değerlerini ayarlayın
hermes gateway      # Mesajlaşma gateway'ini başlatın (Telegram, Discord, vb.)
hermes setup        # Tam kurulum sihirbazını çalıştırın (her şeyi bir kerede yapılandırır)
hermes claw migrate # OpenClaw'dan geçiş yapın (OpenClaw'dan geliyorsanız)
hermes update       # En son sürüme güncelleyin
hermes doctor       # Sorunları teşhis edin
```

📖 **[Tam dökümantasyon →](https://hermes-agent.nousresearch.com/docs/)**

---

## API-anahtarı toplamayı atlayın — Nous Portal

Hermes istediğiniz sağlayıcıyla çalışır — bu değişmeyecek. Ancak model, web araması, görüntü oluşturma, TTS ve bir bulut tarayıcı için beş ayrı API anahtarı toplamak istemiyorsanız, **[Nous Portal](https://portal.nousresearch.com)** hepsini tek bir abonelik altında kapsar:

- **300'den fazla model** — `/model <ad>` ile herhangi birini seçin
- **Tool Gateway** — web araması (Firecrawl), görüntü oluşturma (FAL), metin-konuşma (OpenAI), bulut tarayıcı (Browser Use), hepsi aboneliğiniz üzerinden yönlendirilir. Ek hesap gerekmez.

Yeni bir kurulumdan tek komut:

```bash
hermes setup --portal
```

Bu, sizi OAuth ile giriş yapar, Nous'u sağlayıcınız olarak ayarlar ve Tool Gateway'i etkinleştirir. Herhangi bir zamanda `hermes portal info` ile nelerin bağlı olduğunu kontrol edin. Tam ayrıntılar [Tool Gateway dokümantasyon sayfasında](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

İstediğiniz zaman araç başına kendi anahtarlarınızı kullanmaya devam edebilirsiniz — gateway arka uç bazındadır, ya hep ya hiç değil.

---

## CLI ve Mesajlaşma Hızlı Referansı

Hermes'in iki giriş noktası vardır: terminal UI'ını `hermes` ile başlatın veya gateway'i çalıştırıp Telegram, Discord, Slack, WhatsApp, Signal veya E-posta'dan konuşun. Bir konuşmaya başladığınızda, birçok slash komutu her iki arayüzde de paylaşılır.

| Eylem                            | CLI                                           | Mesajlaşma platformları                                                           |
| -------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------- |
| Sohbete başlama                  | `hermes`                                      | `hermes gateway setup` + `hermes gateway start` çalıştırın, ardından bot'a mesaj gönderin |
| Yeni konuşma başlatma            | `/new` veya `/reset`                          | `/new` veya `/reset`                                                              |
| Model değiştirme                 | `/model [sağlayıcı:model]`                    | `/model [sağlayıcı:model]`                                                        |
| Kişilik belirleme                | `/personality [ad]`                           | `/personality [ad]`                                                               |
| Son turu yeniden dene/geri al    | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                |
| Bağlamı sıkıştır / kullanımı gör | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [gün]`                                          |
| Becerilere göz at                | `/skills` veya `/<beceri-adı>`                | `/<beceri-adı>`                                                                  |
| Mevcut çalışmayı kes             | `Ctrl+C` veya yeni mesaj gönderin             | `/stop` veya yeni mesaj gönderin                                                  |
| Platforma özel durum             | `/platforms`                                  | `/status`, `/sethome`                                                             |

Tam komut listeleri için [CLI rehberine](https://hermes-agent.nousresearch.com/docs/user-guide/cli) ve [Mesajlaşma Gateway rehberine](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) bakın.

---

## Dökümantasyon

Tüm dökümantasyon **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)** adresinde:

| Bölüm                                                                                              | Kapsanan Konular                                             |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [Hızlı Başlangıç](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)           | Kurulum → yapılandırma → 2 dakikada ilk konuşma              |
| [CLI Kullanımı](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                         | Komutlar, tuş bağlantıları, kişilikler, oturumlar            |
| [Yapılandırma](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                | Yapılandırma dosyası, sağlayıcılar, modeller, tüm seçenekler |
| [Mesajlaşma Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)              | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant   |
| [Güvenlik](https://hermes-agent.nousresearch.com/docs/user-guide/security)                         | Komut onayı, DM eşleştirme, konteyner izolasyonu             |
| [Araçlar ve Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)        | 40+ araç, toolset sistemi, terminal arka uçları              |
| [Beceri Sistemi](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)            | Prosedürel bellek, Skills Hub, beceri oluşturma              |
| [Bellek](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                    | Kalıcı bellek, kullanıcı profilleri, en iyi uygulamalar      |
| [MCP Entegrasyonu](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)             | Genişletilmiş yetenekler için herhangi bir MCP sunucusunu bağlayın |
| [Cron Zamanlama](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)              | Platform teslimatlı zamanlanmış görevler                     |
| [Bağlam Dosyaları](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)   | Her konuşmayı şekillendiren proje bağlamı                    |
| [Mimari](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)                  | Proje yapısı, ajan döngüsü, ana sınıflar                     |
| [Katkıda Bulunma](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)         | Geliştirme kurulumu, PR süreci, kod stili                    |
| [CLI Referansı](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                 | Tüm komutlar ve flag'ler                                     |
| [Ortam Değişkenleri](https://hermes-agent.nousresearch.com/docs/reference/environment-variables)   | Tam ortam değişkeni referansı                                |

---

## OpenClaw'dan Geçiş

OpenClaw'dan geliyorsanız, Hermes ayarlarınızı, anılarınızı, becerilerinizi ve API anahtarlarınızı otomatik olarak içe aktarabilir.

**İlk kurulum sırasında:** Kurulum sihirbazı (`hermes setup`) otomatik olarak `~/.openclaw` dizinini algılar ve yapılandırma başlamadan önce geçiş yapmayı teklif eder.

**Kurulumdan sonra herhangi bir zamanda:**

```bash
hermes claw migrate              # Etkileşimli geçiş (tüm preset)
hermes claw migrate --dry-run    # Nelerin taşınacağını önizleyin
hermes claw migrate --preset user-data   # Sırlar olmadan geçiş yapın
hermes claw migrate --overwrite  # Varolan çakışmaların üzerine yazın
```

Neler içe aktarılır:

- **SOUL.md** — kişilik dosyası
- **Anılar** — MEMORY.md ve USER.md girdileri
- **Beceriler** — kullanıcı tarafından oluşturulmuş beceriler → `~/.hermes/skills/openclaw-imports/`
- **Komut beyaz listesi** — onay kalıpları
- **Mesajlaşma ayarları** — platform yapılandırmaları, izin verilen kullanıcılar, çalışma dizini
- **API anahtarları** — beyaz listedeki sırlar (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS varlıkları** — çalışma alanı ses dosyaları
- **Çalışma alanı talimatları** — AGENTS.md (`--workspace-target` ile)

Tüm seçenekler için `hermes claw migrate --help` komutuna bakın veya kuru çalıştırma önizlemeleriyle etkileşimli ajan rehberli geçiş için `openclaw-migration` becerisini kullanın.

---

## Katkıda Bulunma

Katkılarınızı memnuniyetle karşılıyoruz! Geliştirme kurulumu, kod stili ve PR süreci için [Katkıda Bulunma Rehberi'ne](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) bakın.

Katkıda bulunanlar için hızlı başlangıç — standart yükleyiciyi kullanın, ardından oluşturduğu tam git checkout'undan (`$HERMES_HOME/hermes-agent`, genellikle `~/.hermes/hermes-agent`) çalışın. Bu, `hermes update`, yönetilen venv, tembel bağımlılıklar, gateway ve dokümantasyon araçları tarafından kullanılan düzen ile eşleşir.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Manuel klon alternatifi (yönetilen kurulum düzenini istemediğiniz tek kullanımlık klonlar/CI için):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Topluluk

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Hatalar](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — AT-SPI erişilebilirlik ağaçları, Wayland/X11 girişi, ekran görüntüleri ve kompozitör pencere hedefleme ile Hermes ve diğer MCP ana bilgisayarları için Linux masaüstü-kontrol MCP sunucusu.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Topluluk WeChat köprüsü: Aynı WeChat hesabında Hermes Agent ve OpenClaw'u çalıştırın.

---

## Lisans

MIT — bkz. [LICENSE](LICENSE).

[Nous Research](https://nousresearch.com) tarafından geliştirilmiştir.
