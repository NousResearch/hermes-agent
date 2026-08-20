<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Dokümantasyon-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Dokümantasyon"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/Lisans-MIT-green?style=for-the-badge" alt="Lisans: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Geliştiren-Nous%20Research-blueviolet?style=for-the-badge" alt="Geliştiren: Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**[Nous Research](https://nousresearch.com) tarafından geliştirilen, kendini geliştiren yapay zekâ ajanı.** Hermes, yerleşik öğrenme döngüsüyle öne çıkan tek ajandır: deneyimlerinden beceriler oluşturur, bu becerileri kullanım sırasında geliştirir, bilgiyi kalıcılaştırmak için kendine hatırlatmalar yapar, kendi geçmiş konuşmalarında arama yapar ve oturumlar boyunca sizi giderek daha iyi tanıyan bir model oluşturur. 5 dolarlık bir VPS’te, GPU kümesinde veya boştayken neredeyse hiç maliyet oluşturmayan serverless altyapıda çalıştırabilirsiniz. Dizüstü bilgisayarınıza bağlı değildir — ajan bir bulut VM’de çalışırken siz onunla Telegram üzerinden konuşabilirsiniz.

İstediğiniz modeli kullanın — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, kendi endpoint’iniz veya [diğer sağlayıcılar](https://hermes-agent.nousresearch.com/docs/integrations/providers). Sağlayıcıyı ya da modeli `hermes model` ile değiştirebilirsiniz — koda dokunmadan, tek bir sağlayıcıya bağımlı kalmadan.

<table>
<tr><td><b>Gerçek terminal arayüzü</b></td><td>Çok satırlı düzenleme desteği, slash command otomatik tamamlama, konuşma geçmişi, devam eden işi kesip yeniden yönlendirme ve araç çıktısını canlı akış olarak gösterme özelliklerine sahip tam bir TUI.</td></tr>
<tr><td><b>Kullandığınız yerde çalışır</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal ve CLI — hepsi tek bir gateway process’i üzerinden. Sesli notların yazıya dökülmesi ve platformlar arası konuşma sürekliliği desteklenir.</td></tr>
<tr><td><b>Kapalı döngülü öğrenme</b></td><td>Periyodik hatırlatmalarla desteklenen, ajan tarafından düzenlenen bellek. Karmaşık görevlerden sonra otonom beceri oluşturma. Kullanım sırasında kendini geliştiren beceriler. Oturumlar arası hatırlama için LLM özetleriyle desteklenen FTS5 oturum araması. <a href="https://github.com/plastic-labs/honcho">Honcho</a> ile diyalektik kullanıcı modellemesi. <a href="https://agentskills.io">agentskills.io</a> açık standardı ile uyumlu.</td></tr>
<tr><td><b>Zamanlanmış otomasyonlar</b></td><td>Çıktıları herhangi bir platforma gönderebilen yerleşik cron scheduler. Günlük raporlar, gecelik yedeklemeler, haftalık denetimler — hepsi doğal dille tanımlanır ve gözetimsiz çalışır.</td></tr>
<tr><td><b>Görevleri devreder ve paralel yürütür</b></td><td>Paralel iş akışları için izole subagent’lar oluşturun. RPC üzerinden araçları çağıran Python script’leri yazın; çok adımlı pipeline’ları bağlam maliyeti oluşturmayan adımlara dönüştürün.</td></tr>
<tr><td><b>Sadece dizüstünüzde değil, her yerde çalışır</b></td><td>Altı terminal backend’i — local, Docker, SSH, Singularity, Modal ve Daytona. Daytona ve Modal serverless persistence sunar: ajanınızın ortamı boştayken hibernation moduna geçer, talep üzerine uyanır ve oturumlar arasında neredeyse hiç maliyet oluşturmaz. 5 dolarlık bir VPS’te veya GPU kümesinde çalıştırabilirsiniz.</td></tr>
<tr><td><b>Araştırmaya hazır</b></td><td>Batch trajectory üretimi ve yeni nesil tool-calling modelleri eğitmek için trajectory compression.</td></tr>
</table>

---

## Hızlı Kurulum

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (native, PowerShell)

> **Önemli Not:** Hermes Windows’ta WSL olmadan native çalışır — CLI, gateway, TUI ve araçların tamamı native olarak desteklenir. WSL2 kullanmayı tercih ederseniz yukarıdaki Linux/macOS tek satırlık komut orada da çalışır. Bir hata mı buldunuz? Lütfen [issue açın](https://github.com/NousResearch/hermes-agent/issues).

Bunu PowerShell’de çalıştırın:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Installer gerekli her şeyi hazırlar: uv, Python 3.11, Node.js, ripgrep, ffmpeg ve **portable Git Bash** (MinGit, `%LOCALAPPDATA%\hermes\git` dizinine açılır — yönetici yetkisi gerekmez, sistemdeki Git kurulumlarından tamamen izoledir). Hermes, shell komutlarını çalıştırmak için pakete dahil bu Git Bash’i kullanır.

Sistemde Git zaten kuruluysa installer bunu algılar ve onu kullanır. Aksi takdirde yaklaşık 45 MB’lık MinGit indirmesi yeterlidir — sistemdeki Git kurulumunuza dokunmaz veya müdahale etmez.

> **Android / Termux:** Test edilmiş manuel kurulum yolu [Termux kılavuzunda](https://hermes-agent.nousresearch.com/docs/getting-started/termux) belgelenmiştir. Termux’ta Hermes, özel olarak hazırlanmış `.[termux]` extra’sını kurar; çünkü tam `.[all]` extra’sı şu anda Android ile uyumsuz ses bağımlılıkları içerir.
>
> **Windows:** Native Windows tam olarak desteklenir — yukarıdaki PowerShell komutu gerekli her şeyi kurar. WSL2 kullanmayı tercih ederseniz Linux komutu orada da çalışır. Native Windows kurulumu `%LOCALAPPDATA%\hermes` altındadır; WSL2 kurulumu ise Linux’taki gibi `~/.hermes` altına yapılır.

Kurulumdan sonra:

```bash
source ~/.bashrc    # shell’i yeniden yükleyin (veya: source ~/.zshrc)
hermes              # sohbete başlayın!
```

### Sorun Giderme

#### Windows Defender veya antivirüs `uv.exe` dosyasını zararlı olarak işaretliyor

Antivirüs yazılımınız (Bitdefender, Windows Defender vb.) Hermes `bin` klasöründeki (`%LOCALAPPDATA%\hermes\bin\uv.exe`) `uv.exe` dosyasını karantinaya alırsa, bu büyük olasılıkla **false positive / hatalı pozitif** durumudur. Bu dosya Astral’ın `uv` aracıdır — Hermes’in Python ortamını yönetmek için pakete dahil ettiği Rust tabanlı Python paket yöneticisidir (package manager). Makine öğrenmesi tabanlı antivirüs motorları, paket indirip kuran imzasız Rust binary’lerini sık sık işaretleyebilir.

**Kopyanızın orijinal olduğunu doğrulamak için:**

```powershell
# Gerekirse GitHub CLI kurun
winget install --id GitHub.cli

# GitHub’a giriş yapın
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

Doğrulama işlemi (attestation verify) "Verification succeeded" diyorsa ve son satır `True` olarak görünüyorsa sorun yoktur.

**Hermes’i allowlist’e eklemek için:**
- **Windows Defender:** PowerShell’i Yönetici olarak çalıştırın → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Bitdefender konsolunda istisna ekleyin (Protection > Antivirus > Settings > Manage Exceptions)
- Dosya hash’ini değil, **klasörü** allowlist’e ekleyin — Hermes `uv` aracını günceller ve hash her sürümde değişir

Daha fazla bilgi için Astral tarafındaki upstream issue’lara bakın: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Başlarken

```bash
hermes              # Etkileşimli CLI — bir konuşma başlatın
hermes model        # LLM sağlayıcınızı ve modelinizi seçin
hermes tools        # Hangi araçların etkin olduğunu yapılandırın
hermes config set   # Belirli yapılandırma değerlerini ayarlayın
hermes gateway      # Messaging Gateway’i başlatın (Telegram, Discord vb.)
hermes setup        # Tam kurulum sihirbazını çalıştırın (her şeyi tek seferde yapılandırır)
hermes claw migrate # OpenClaw’dan geçiş yapın (OpenClaw’dan geliyorsanız)
hermes update       # En son sürüme güncelleyin
hermes doctor       # Sorunları teşhis edin
```

📖 **[Tüm dokümantasyon →](https://hermes-agent.nousresearch.com/docs/)**

---

## Ayrı API anahtarlarıyla uğraşmayın — Nous Portal

Hermes istediğiniz sağlayıcıyla çalışır — bu değişmeyecek. Ancak model, web araması, görüntü oluşturma, TTS ve bulut tarayıcı (cloud browser) için beş farklı API anahtarını tek tek toplamakla uğraşmak istemiyorsanız, **[Nous Portal](https://portal.nousresearch.com)** hepsini tek bir abonelik altında kapsar:

- **300’den fazla model** — herhangi birini `/model <name>` ile seçin
- **Tool Gateway** — web araması (Firecrawl), görüntü oluşturma (FAL), TTS / metinden sese (OpenAI), cloud browser (Browser Use); tamamı aboneliğiniz üzerinden yönlendirilir. Ek hesap gerekmez.

Temiz kurulumdan sonra tek komut yeter:

```bash
hermes setup --portal
```

Bu komut OAuth üzerinden giriş yapmanızı sağlar, Nous’u sağlayıcınız olarak ayarlar ve Tool Gateway’i etkinleştirir. Hangi entegrasyonların bağlı olduğunu istediğiniz zaman `hermes portal info` ile kontrol edebilirsiniz. Tüm ayrıntılar [Tool Gateway dokümantasyon sayfasında](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway) yer alır.

İstediğiniz zaman araç bazında kendi anahtarlarınızı kullanmaya devam edebilirsiniz — gateway backend bazında yapılandırılır; ya hep ya hiç değildir.

---

## CLI ve Messaging için Hızlı Referans

Hermes’in iki giriş noktası vardır: `hermes` ile terminal UI’ı başlatabilir veya gateway’i çalıştırıp Telegram, Discord, Slack, WhatsApp, Signal ya da E-posta üzerinden konuşabilirsiniz. Bir konuşmaya girdikten sonra birçok slash command her iki arayüzde de ortaktır.

| Eylem                                | CLI                                           | Messaging platformları                                                           |
| ------------------------------------ | --------------------------------------------- | --------------------------------------------------------------------------------- |
| Sohbet başlat                        | `hermes`                                      | `hermes gateway setup` + `hermes gateway start` çalıştırın, ardından bota mesaj gönderin |
| Yeni konuşma başlat                  | `/new` veya `/reset`                          | `/new` veya `/reset`                                                              |
| Model değiştir                       | `/model [provider:model]`                     | `/model [provider:model]`                                                         |
| Personality ayarla                   | `/personality [name]`                         | `/personality [name]`                                                             |
| Son adımı yeniden dene veya geri al  | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                 |
| Context’i sıkıştır / kullanım bilgisini gör   | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                         |
| Becerilere göz at                    | `/skills` veya `/<skill-name>`                | `/<skill-name>`                                                                   |
| Devam eden işi kes                   | `Ctrl+C` veya yeni bir mesaj gönderin        | `/stop` veya yeni bir mesaj gönderin                                              |
| Platforma özgü durum                 | `/platforms`                                  | `/status`, `/sethome`                                                             |

Tam komut listeleri için [CLI kılavuzuna](https://hermes-agent.nousresearch.com/docs/user-guide/cli) ve [Messaging Gateway kılavuzuna](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) bakın.

---

## Dokümantasyon

Tüm dokümantasyon **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)** adresindedir:

| Bölüm                                                                                              | İçerik                                                       |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [Hızlı Başlangıç](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)           | Kurulum → yapılandırma → 2 dakikada ilk konuşma             |
| [CLI Kullanımı](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                          | Komutlar, kısayol tuşları, personality’ler, oturumlar        |
| [Yapılandırma](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                | Yapılandırma dosyası, sağlayıcılar, modeller ve tüm seçenekler |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)               | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant   |
| [Güvenlik](https://hermes-agent.nousresearch.com/docs/user-guide/security)                          | Komut onayı, DM eşleştirme, container izolasyonu             |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)            | 40’tan fazla araç, toolset sistemi, terminal backend’leri    |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)              | Prosedürel bellek, Skills Hub, beceri oluşturma              |
| [Bellek](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                     | Kalıcı bellek, kullanıcı profilleri, en iyi uygulamalar      |
| [MCP Entegrasyonu](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)              | Genişletilmiş yetenekler için herhangi bir MCP sunucusuna bağlanma |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)              | Platformlara gönderim desteğiyle zamanlanmış görevler        |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)       | Her konuşmayı şekillendiren proje context’i                  |
| [Mimari](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)                   | Proje yapısı, agent loop, temel sınıflar                     |
| [Katkıda Bulunma](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)          | Geliştirme ortamı kurulumu, PR süreci, kod stili             |
| [CLI Referansı](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                  | Tüm komutlar ve flag’ler                                     |
| [Ortam Değişkenleri](https://hermes-agent.nousresearch.com/docs/reference/environment-variables)   | Ortam değişkenlerinin tam referansı                          |

---

## OpenClaw’dan Geçiş

OpenClaw’dan geliyorsanız Hermes ayarlarınızı, belleklerinizi, becerilerinizi ve API anahtarlarınızı otomatik olarak içe aktarabilir.

**İlk kurulum sırasında:** Kurulum sihirbazı (`hermes setup`) `~/.openclaw` dizinini otomatik olarak algılar ve yapılandırma başlamadan önce geçiş yapmayı önerir.

**Kurulumdan sonra istediğiniz zaman:**

```bash
hermes claw migrate              # Etkileşimli geçiş (full preset)
hermes claw migrate --dry-run    # Nelerin taşınacağının önizlemesi
hermes claw migrate --preset user-data   # Secrets olmadan geçiş
hermes claw migrate --overwrite  # Mevcut çakışmaların üzerine yaz
```

İçe aktarılanlar:

- **SOUL.md** — persona dosyası
- **Bellekler** — MEMORY.md ve USER.md girdileri
- **Beceriler** — kullanıcı tarafından oluşturulan beceriler → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — onay kalıpları
- **Messaging ayarları** — platform yapılandırmaları, izinli kullanıcılar, çalışma dizini
- **API anahtarları** — allowlist’teki secrets / gizli değerler (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS asset’leri** — çalışma alanındaki ses dosyaları
- **Workspace talimatları** — AGENTS.md (`--workspace-target` ile)

Tüm seçenekler için `hermes claw migrate --help` komutuna bakın veya dry-run önizlemeleri içeren, ajan rehberli etkileşimli geçiş için `openclaw-migration` becerisini kullanın.

---

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Geliştirme ortamı kurulumu, kod stili ve PR süreci için [Katkıda Bulunma Kılavuzuna](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) bakın.

Katkıda bulunanlar için hızlı başlangıç — standart installer’ı kullanın, ardından installer’ın `$HERMES_HOME/hermes-agent` altında oluşturduğu tam Git checkout’u üzerinde çalışın (genellikle `~/.hermes/hermes-agent`). Bu layout, `hermes update`, managed venv, lazy dependency sistemi, gateway ve docs tooling tarafından kullanılan yapıyla aynıdır.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Manuel klonlama alternatifi (managed install layout’u özellikle istemediğiniz geçici klonlar/CI ortamları için):

Venv’i proje klasörünün dışında oluşturun. Ajanın çalıştığı klasörün içindeki bir venv, kendi checkout’u üzerinde göreli yol kullanan bir komutla silinebilir; bu da oturum sırasında çalışan ortamı bozabilir.

```bash
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
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Hermes ve diğer MCP host’ları için Linux masaüstü kontrolü sağlayan MCP sunucusu; AT-SPI accessibility tree’leri, Wayland/X11 input, screenshot desteği ve compositor’da pencere hedefleme sunar.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Topluluk tarafından geliştirilen WeChat köprüsü: Hermes Agent ve OpenClaw'u aynı WeChat hesabında çalıştırın.

---

## Lisans

MIT — [LICENSE](LICENSE) dosyasına bakın.

[Nous Research](https://nousresearch.com) tarafından geliştirilmiştir.
