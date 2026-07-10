# Hermes Agent'e Katkıda Bulunmak

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python" alt="Python 3.11+">
</p>

Hermes Agent'e katkıda bulunduğunuz için teşekkürler! Bu rehber, geliştirme ortamınızı kurmaktan mimariyi anlamaya, ne inşa edeceğinize karar vermekten PR'ınızın birleştirilmesine kadar her şeyi kapsar.

---

## Katkı Öncelikleri

Katkıları bu sırayla değerlendiriyoruz:

1. **Hata düzeltmeleri** — çökmeler, yanlış davranış, veri kaybı. Her zaman en yüksek öncelik.
2. **Çapraz platform uyumluluğu** — macOS, farklı Linux dağıtımları ve Windows üzerinde WSL2. Hermes'in her yerde çalışmasını istiyoruz.
3. **Güvenlik sağlamlaştırması** — shell injection, prompt injection, path traversal, privilege escalation. Bkz. [Güvenlik Değerlendirmeleri](#güvenlik-değerlendirmeleri).
4. **Performans ve sağlamlık** — yeniden deneme mantığı, hata yönetimi, zarif düşüş.
5. **Yeni yetenekler** — ancak yalnızca geniş çapta kullanışlı olanlar. Bkz. [Yetenek mi, Araç mı?](#yetenek-mi-araç-mı)
6. **Yeni araçlar** — nadiren gerekir. Çoğu yetenek yetenek (skill) olmalıdır.
7. **Dokümantasyon** — düzeltmeler, açıklamalar, yeni örnekler.

---

## Başlamadan Önce: Önce Arayın

Hızlı bir arama zaman kazandırır ve PR kuyruğunu temiz tutar — tekrarlar yaygındır.

- **Açık ve birleştirilmiş PR/issue'ları arayın**:
  ```bash
  gh search issues --repo NousResearch/hermes-agent "<terimleriniz>"
  gh search prs --repo NousResearch/hermes-agent --state all "<terimleriniz>"
  ```
  Veya web: [issues](https://github.com/NousResearch/hermes-agent/issues?q=) · [PRs](https://github.com/NousResearch/hermes-agent/pulls?q=is%3Apr).
- **Issue tracker kodun gerisinde kalabilir** — kaynak kodunda da arama yapın.
- **Açık PR varsa**, onu inceleyin, rakip kopya açmayın.
- **Büyük işler için** issue'ya yorum yaparak başladığınızı belirtin.

---

## Yetenek mi, Araç mı?

Bu, yeni katkıda bulunanlar için en yaygın sorudur. Cevap neredeyse her zaman **yetenek (skill)**.

### Yetenek (Skill) yapın:

- Yetenek, talimatlar + shell komutları + mevcut araçlar olarak ifade edilebiliyorsa
- Ajanın `terminal` veya `web_extract` ile çağırabileceği harici bir CLI veya API'yi sarıyorsa
- Özel Python entegrasyonu veya API anahtarı yönetimi gerektirmiyorsa
- Örnekler: arXiv araması, git iş akışları, Docker yönetimi, PDF işleme, CLI araçları ile e-posta

### Araç (Tool) yapın:

- API anahtarları, kimlik doğrulama veya çok bileşenli yapılandırma ile uçtan uca entegrasyon gerektiriyorsa
- Her seferinde hassas yürütülmesi gereken özel işlem mantığı varsa
- Terminal üzerinden gidemeyen ikili veri, akış veya gerçek zamanlı olayları işliyorsa
- Örnekler: tarayıcı otomasyonu (Browserbase), TTS (ses kodlama), görüntü analizi (base64)

### Yetenek Dahil Edilmeli mi?

Dahil edilen yetenekler (`skills/`), her Hermes kurulumuyla birlikte gelir ve **çoğu kullanıcı için geniş çapta kullanışlı** olmalıdır. Yeteneğiniz resmi ancak herkes için gerekli değilse (ör. ücretli bir servis, ağır bağımlılık), `optional-skills/` dizinine koyun. Kullanıcılar `hermes skills browse` ile keşfedebilir ve `hermes skills install` ile kurabilir.

Uzmanlaşmış, topluluk katkılı veya niş yetenekler için **Skills Hub**'ı kullanın — [Nous Research Discord](https://discord.gg/NousResearch) üzerinde paylaşın.

---

## Bellek Sağlayıcıları: Bağımsız Eklenti Olarak Yayınlayın

**Artık bu depoya yeni bellek sağlayıcıları kabul etmiyoruz.** `plugins/memory/` altındaki yerleşik sağlayıcılar (honcho, mem0, supermemory, byterover, hindsight, holographic, openviking, retaindb) kapatılmıştır. Yeni bir bellek arka ucu için bağımsız bir eklenti deposu yayınlayın.

Bağımsız bellek eklentileri: `MemoryProvider` ABC'sini (`agent/memory_provider.py`) uygular, `discover_memory_providers()` ile keşfedilir, `hermes memory setup` ile entegre olur, `register_cli(subparser)` ile CLI komutları ekleyebilir.

`plugins/memory/` altında yeni dizin ekleyen PR'ler kapatılacaktır.

---

## Geliştirme Ortamı

| Gereksinim | Not |
|------------|-----|
| **Git** + `git-lfs` | |
| **Python 3.11+** | uv eksikse yükler |
| **uv** | [Kurulum](https://docs.astral.sh/uv/) |
| **Node.js 20+** | İsteğe bağlı (tarayıcı araçları, WhatsApp) |

### Kurulum

Standart yükleyici (önerilen):
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
npm install  # isteğe bağlı
```

Manuel klonlama:
```bash
git clone https://github.com/NousResearch/hermes-agent.git && cd hermes-agent
uv venv venv --python 3.11 && export VIRTUAL_ENV="$(pwd)/venv"
uv pip install -e ".[all,dev]"
```

### Yapılandırma

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env
echo "OPENROUTER_API_KEY=***" >> ~/.hermes/.env
```

### Çalıştırma

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes
hermes doctor
hermes chat -q "Merhaba"
```

### Testler

```bash
scripts/run_tests.sh        # CI ile uyumlu (önerilen)
pytest tests/ -v            # alternatif (venv etkinken)
```

---

## Proje Yapısı

```
hermes-agent/
├── run_agent.py              # AIAgent — konuşma döngüsü, araç dağıtımı
├── cli.py                    # HermesCLI — etkileşimli TUI
├── model_tools.py            # Araç orkestrasyonu
├── toolsets.py               # Araç gruplamaları
├── hermes_state.py           # SQLite oturum veritabanı
├── batch_runner.py           # Paralel toplu işleme
│
├── agent/                    # Ajan iç işleyişi
│   ├── prompt_builder.py, context_compressor.py
│   ├── auxiliary_client.py, display.py
│   ├── model_metadata.py, trajectory.py
│
├── hermes_cli/               # CLI komutları
│   ├── main.py, config.py, setup.py, auth.py
│   ├── models.py, banner.py, commands.py
│   ├── callbacks.py, doctor.py, skills_hub.py
│   └── skin_engine.py
│
├── tools/                    # Araç uygulamaları (kendi kendine kaydolan)
│   ├── registry.py, approval.py, terminal_tool.py
│   ├── file_operations.py, web_tools.py
│   ├── vision_tools.py, delegate_tool.py
│   ├── code_execution_tool.py, session_search_tool.py
│   ├── cronjob_tools.py, skill_tools.py
│   └── environments/ (local, docker, ssh, modal, vb.)
│
├── gateway/                  # Mesajlaşma ağ geçidi
│   ├── run.py, config.py, session.py
│   └── platforms/ (telegram, discord, slack, whatsapp)
│
├── scripts/                  # Yükleyici betikleri
│   ├── install.sh, install.ps1
│   └── whatsapp-bridge/
│
├── skills/                   # Dahil edilen yetenekler
├── optional-skills/          # İsteğe bağlı yetenekler
├── tests/                    # Test paketi
├── website/                  # Dokümantasyon sitesi
│
├── cli-config.yaml.example
└── AGENTS.md
```

### Kullanıcı yapılandırması (`~/.hermes/`)

| Yol | Amaç |
|-----|------|
| `~/.hermes/config.yaml` | Ayarlar (model, terminal, toolsets, vb.) |
| `~/.hermes/.env` | API anahtarları ve sırlar |
| `~/.hermes/auth.json` | OAuth kimlik bilgileri |
| `~/.hermes/skills/` | Aktif yetenekler |
| `~/.hermes/memories/` | Kalıcı bellek |
| `~/.hermes/state.db` | SQLite oturum veritabanı |
| `~/.hermes/sessions/` | Ağ geçidi yönlendirme |
| `~/.hermes/cron/` | Zamanlanmış görevler |

---

## Mimariye Genel Bakış

### Ana Döngü

```
Kullanıcı mesajı → AIAgent._run_agent_loop()
  ├── Sistem promptu oluştur (prompt_builder.py)
  ├── API kwargs'larını oluştur (model, mesajlar, araçlar, akıl yürütme)
  ├── LLM'yi çağır (OpenAI uyumlu API)
  ├── Yanıtta tool_calls varsa:
  │     ├── Her aracı kayıt dağıtımı üzerinden yürüt
  │     ├── Araç sonuçlarını konuşmaya ekle
  │     └── LLM çağrısına geri dön
  ├── Metin yanıtı varsa:
  │     ├── Oturumu DB'ye kaydet
  │     └── final_response döndür
  └── Token sınırına yaklaşılıyorsa bağlam sıkıştırma
```

### Temel Tasarım Desenleri

- **Kendi kendine kaydolan araçlar**: Her araç dosyası `registry.register()` çağırır; `model_tools.py` tüm araç modüllerini içe aktararak keşfi tetikler
- **Araç seti gruplaması**: Araçlar `web`, `terminal`, `file`, `browser` gibi gruplara ayrılır, platform başına etkinleştirilip devre dışı bırakılabilir
- **Oturum kalıcılığı**: Tüm konuşmalar SQLite'da (`hermes_state.py`) FTS5 tam metin arama ile saklanır
- **Geçici enjeksiyon**: Sistem promptları ve ön doldurma mesajları API çağrısı anında enjekte edilir, asla kaydedilmez
- **Sağlayıcı soyutlaması**: Ajan herhangi bir OpenAI uyumlu API ile çalışır

## AIAgent Sınıfı

`AIAgent.__init__` yaklaşık 60 parametre alır. Sık dokunacağınız minimum alt küme:

```python
class AIAgent:
    def __init__(self, base_url=None, api_key=None, provider=None,
                 model="", max_iterations=90, enabled_toolsets=None,
                 platform=None, session_id=None, ...): ...
    def chat(self, message: str) -> str: ...
    def run_conversation(self, user_message, ...) -> dict: ...
```

Ajan döngüsü (`run_conversation()` içinde):
```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0):
    if self._interrupt_requested: break
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else:
        return response.content
```

## CLI Mimarisi ve Slash Komutları

- **Rich** başlık/paneller için, **prompt_toolkit** giriş ve otomatik tamamlama için
- **KawaiiSpinner** (`agent/display.py`) — API çağrıları sırasında animasyonlu yüzler
- Yetenek slash komutları `~/.hermes/skills/` dizinini tarar, önbelleği korumak için **kullanıcı mesajı** olarak enjekte eder

### Slash Komut Kaydı (`hermes_cli/commands.py`)

Tüm slash komutları, `COMMAND_REGISTRY` listesindeki `CommandDef` nesneleridir. Her kanal bundan otomatik türetilir: CLI, gateway, Telegram, Slack, otomatik tamamlama.

```python
CommandDef("mycommand", "Açıklama", "Session", aliases=("mc",), args_hint="[arg]"),
```

Ekleme: `COMMAND_REGISTRY`'ye giriş + `cli.py`'de işleyici + gateway'de işleyici.


## Kod Stili

- **PEP 8** pratik istisnalarla (katı satır uzunluğu uygulamıyoruz)
- **Yorumlar**: Yalnızca belirgin olmayan niyeti, ödünleşimleri veya API tuhaflıklarını açıklarken. Kodun ne yaptığını anlatmayın.
- **Hata yönetimi**: Spesifik istisnaları yakalayın. `logger.warning()`/`logger.error()` ile günlükleyin — beklenmeyen hatalar için `exc_info=True` kullanın.
- **Çapraz platform**: Asla Unix varsaymayın. Bkz. [Çapraz Platform Uyumluluğu](#çapraz-platform-uyumluluğu)

---

## Yeni Bir Araç Ekleme

Bir araç yazmadan önce kendinize sorun: [bunun yerine bir yetenek mi olmalı?](#yetenek-mi-araç-mı)

Araçlar merkezi kayda kendi kendine kaydolur. Her araç dosyası, şemasını, işleyicisini ve kaydını birlikte barındırır:

```python
"""my_tool — Bu aracın ne yaptığının kısa açıklaması."""

import json
from tools.registry import registry


def my_tool(param1: str, param2: int = 10, **kwargs) -> str:
    """İşleyici. Genellikle JSON olan bir dize sonucu döndürür."""
    result = do_work(param1, param2)
    return json.dumps(result)


MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Bu aracın ne yaptığı ve ajanın ne zaman kullanması gerektiği.",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "param1 nedir"},
                "param2": {"type": "integer", "description": "param2 nedir", "default": 10},
            },
            "required": ["param1"],
        },
    },
}


def _check_requirements() -> bool:
    """Bu aracın bağımlılıkları mevcutsa True döndür."""
    return True


registry.register(
    name="my_tool",
    toolset="my_toolset",
    schema=MY_TOOL_SCHEMA,
    handler=lambda args, **kw: my_tool(**args, **kw),
    check_fn=_check_requirements,
)
```

**Araç setine bağlayın (zorunlu):** Yerleşik araçlar otomatik keşfedilir — `tools/*.py` içindeki `registry.register()` çağrıları, `model_tools` yüklendiğinde `discover_builtin_tools()` tarafından içe aktarılır. Yine de araç adını `toolsets.py` içindeki uygun listeye (ör. `_HERMES_CORE_TOOLS`) eklemelisiniz; aksi takdirde araç kaydedilir ancak ajana gösterilmez.

---

## Yeni Bir Yetenek (Skill) Ekleme

Dahil edilen yetenekler `skills/` dizininde, isteğe bağlı olanlar `optional-skills/` içinde kategorize edilir:

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Zorunlu: ana talimatlar
│       └── scripts/              # İsteğe bağlı: yardımcı betikler
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

### SKILL.md formatı

```markdown
---
name: my-skill
description: Kısa açıklama (yetenek arama sonuçlarında gösterilir)
version: 1.0.0
author: Adınız
license: MIT
platforms: [macos, linux]
required_environment_variables:
  - name: MY_API_KEY
    prompt: API anahtarı
    help: Nereden alınır
    required_for: tam işlevsellik
prerequisites:
  env_vars: [MY_API_KEY]
  commands: [curl, jq]
metadata:
  hermes:
    tags: [Kategori, Altkategori, Anahtar Kelimeler]
    related_skills: [other-skill-name]
    fallback_for_toolsets: [web]
    requires_toolsets: [terminal]
---

# Yetenek Başlığı

Kısa giriş.

## Ne Zaman Kullanılır
Tetikleme koşulları.

## Ön Koşullar
Env var'ları, kurulum adımları, MCP kurulumu, API anahtarı temini.

## Nasıl Çalıştırılır
`terminal` aracı aracılığıyla kanonik çağrı.

## Hızlı Referans
Yaygın komutlar veya API çağrıları tablosu.

## Prosedür
Ajanın izlediği adım adım talimatlar.

## Bilinen Sorunlar
Bilinen hata modları ve bunların nasıl ele alınacağı.

## Doğrulama
Ajanın çalıştığını nasıl onayladığı.
```

### Platforma özgü yetenekler

Yetenekler, `platforms` ön yüz alanı aracılığıyla hangi işletim sistemlerini desteklediklerini bildirebilir:

```yaml
platforms: [macos]            # yalnızca macOS
platforms: [macos, linux]     # macOS ve Linux
platforms: [windows]          # yalnızca Windows
```

### Koşullu yetenek etkinleştirme

`metadata.hermes` altında dört alan desteklenir:

- `fallback_for_toolsets`: Yalnızca belirtilen araç setleri kullanılamadığında göster
- `requires_toolsets`: Yalnızca belirtilen araç setleri kullanılabildiğinde göster
- `fallback_for_tools`: Yalnızca belirtilen araçlar kullanılamadığında göster
- `requires_tools`: Yalnızca belirtilen araçlar kullanılabildiğinde göster

### Yetenek yazma standartları (ZORUNLU)

Her yeni veya modernize edilmiş yetenek — dahil edilmiş, isteğe bağlı veya katkıda bulunulmuş — birleştirilmeden önce bu standartları karşılamalıdır:

1. **`description` ≤ 60 karakter**, tek cümle, nokta ile biter. Pazarlama kelimeleri yok.
2. **Araç referansları**: Yerel Hermes araçlarını ters tırnakta kullanın: `` `terminal` ``, `` `web_extract` ``, `` `web_search` ``, `` `read_file` ``, `` `write_file` ``, vb.
3. **`platforms:`** alanı gerçek betik bağımlılıklarına göre denetlenmelidir.
4. **`author`** önce insan katkıda bulunanı kredilendirir.
5. **Modern bölüm sırası**: `## Ne Zaman Kullanılır`, `## Ön Koşullar`, `## Nasıl Çalıştırılır`, `## Hızlı Referans`, `## Prosedür`, `## Bilinen Sorunlar`, `## Doğrulama`.
6. **Betikler** `scripts/`, referanslar `references/`, şablonlar `templates/` dizinine gider.
7. **Testler** `tests/skills/test_<skill>_skill.py` konumunda, yalnızca stdlib + pytest.
8. **`.env.example`** eklemeleri ayrı bir blokta izole edilmelidir.

---

## Skin / Tema Ekleme

Hermes, veri odaklı bir skin sistemi kullanır — yeni bir skin eklemek için kod değişikliği gerekmez.

**A. Kullanıcı skin'i (YAML):** `~/.hermes/skins/<ad>.yaml` oluşturun:
```yaml
name: temam
description: Kısa açıklama
colors:
  banner_border: "#HEX"
  banner_title: "#HEX"
  banner_accent: "#HEX"
  banner_dim: "#HEX"
  banner_text: "#HEX"
  response_border: "#HEX"
spinner:
  waiting_faces: ["(⚔)", "(⛨)"]
  thinking_faces: ["(⚔)", "(⌁)"]
  thinking_verbs: ["dövüyor", "planlıyor"]
branding:
  agent_name: "Ajanım"
  welcome: "Hoş geldiniz"
  response_label: " ⚔ Ajan "
  prompt_symbol: "⚔"
tool_prefix: "╎"
```
Tüm alanlar isteğe bağlıdır — eksik değerler varsayılandan devralınır.

**B. Yerleşik skin:** `hermes_cli/skin_engine.py` içindeki `_BUILTIN_SKINS` sözlüğüne ekleyin.

**Aktifleştirme:** `/skin temam` (CLI) veya `display.skin: temam` (config.yaml)

---

## Çapraz Platform Uyumluluğu

Hermes Linux, macOS ve Windows (WSL2 dahil) üzerinde çalışır.

> **PR öncesi:** `scripts/check-windows-footguns.py` çalıştırın.

### Kritik kurallar

1. **`os.kill(pid, 0)` KULLANMAYIN** — Windows'ta sessiz öldürmedir. `psutil.pid_exists(pid)` kullanın.
2. **`shutil.which()` kullanın** — `ps`, `kill`, `grep` Windows'ta bulunmaz.
3. **`termios`/`fcntl`** yalnızca Unix. `ImportError` + `NotImplementedError` yakalayın.
4. **Dosya kodlaması:** `.env` `cp1252` olabilir. `encoding="utf-8-sig"` kullanın.
5. **Süreç yönetimi:** `os.setsid()`, `os.killpg()` Windows'ta farklıdır. `psutil` kullanın.
6. **Windows'ta olmayan sinyaller:** `SIGALRM`, `SIGCHLD`, `SIGHUP`, `SIGUSR1/2`, `SIGPIPE`, `SIGQUIT`, `SIGKILL`.
7. **Yol ayırıcıları:** `pathlib.Path` kullanın.
8. **Sembolik bağlantılar** Windows'ta yükseltilmiş ayrıcalık gerektirir.
9. **POSIX dosya modları** NTFS'de uygulanmaz.
10. **Arka plan daemonları:** `pythonw.exe` gerekir, `python.exe` değil.
11. **Yükleyiciler:** `install.sh` değişikliğini `install.ps1`'e de yansıtın.

---

## Güvenlik Değerlendirmeleri

Hermes terminal erişimine sahiptir. Güvenlik önemlidir.

### Mevcut korumalar

| Katman | Uygulama |
|--------|----------|
| **Sudo şifre iletimi** | Shell enjeksiyonunu önlemek için `shlex.quote()` kullanır |
| **Tehlikeli komut tespiti** | `tools/approval.py` içinde regex desenleri + kullanıcı onay akışı |
| **Cron prompt enjeksiyonu** | `tools/cronjob_tools.py` tarayıcı, talimat geçersiz kılma desenlerini engeller |
| **Yazma kara listesi** | `os.path.realpath()` ile sembolik bağlantı atlatmasını önleyen korumalı yollar |
| **Skills guard** | Hub'dan yüklenen yetenekler için güvenlik tarayıcısı (`tools/skills_guard.py`) |
| **Kod yürütme sandbox'ı** | `execute_code` alt süreci API anahtarları temizlenmiş ortamda çalışır |
| **Konteyner sağlamlaştırması** | Docker: tüm yetenekler düşürülmüş, ayrıcalık yok, PID sınırları, tmpfs |

### Güvenlik hassasiyeti olan kod katkısı yaparken

- Kullanıcı girdisini shell komutlarına yerleştirirken **her zaman `shlex.quote()` kullanın**
- Yol tabanlı erişim kontrolünden önce sembolik bağlantıları `os.path.realpath()` ile çözümleyin
- **Sırları günlüğe kaydetmeyin.** API anahtarları, token'lar ve şifreler asla günlük çıktısında görünmemelidir
- Tek bir hatanın ajan döngüsünü çökertmemesi için araç yürütme etrafında geniş istisnalar yakalayın
- Değişikliğiniz dosya yollarına, süreç yönetimine veya shell komutlarına dokunuyorsa **tüm platformlarda test edin**

### Bağımlılık sabitleme politikası (tedarik zinciri sağlamlaştırması)

Mart 2026'daki [litellm](https://github.com/BerriAI/litellm/issues/24512) ve Mayıs 2026'daki [Mini Shai-Hulud](https://socket.dev/blog/tanstack-npm-packages-compromised-mini-shai-hulud-supply-chain-attack) saldırıları sonrası tüm bağımlılıklar şu kurallara uymalıdır:

| Kaynak | Gereken | Gerekçe |
|--------|---------|---------|
| **PyPI** | `>=taban,<sonraki_major` | Sürümler değişmez ancak aralığa yenileri itilebilir |
| **Git URL** | Tam commit SHA | Dallar değişebilir; SHA içerik adreslidir |
| **GitHub Actions** | SHA + sürüm yorumu | Etiketler değişebilir |
| **CI pip** | `==tam` | Hermetik derlemeler |

**Her PyPI bağımlılığı `<sonraki_major` üst sınıra sahip olmalıdır.** Sınırsız `>=X.Y.Z` reddedilir. 1.x için `<2`, 0.x için `<0.(mevcut_minör + 2)` kullanın.

---

## Test

**HER ZAMAN `scripts/run_tests.sh` kullanın** — doğrudan `pytest` çağırmayın. Betik, CI ile ortam paritesini zorlar (API anahtarları temizlenir, TZ=UTC, `-n auto` xdist).

```bash
scripts/run_tests.sh                                  # tüm paket
scripts/run_tests.sh tests/gateway/                   # tek dizin
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # tek test
```

Her test, `tests/_isolate_plugin.py` ile izole bir alt süreçte çalışır. Hata ayıklama için `--no-isolate` kullanın.

### Değişim-dedektörü testleri yazmayın

Model katalogları veya sürüm numaraları gibi güncellenmesi beklenen verilerin anlık görüntüsünü test etmeyin. Bunun yerine davranışsal değişmezleri (invariant) test edin:

```python
# Kötü — güncellemeyle kırılır
assert len(_PROVIDER_MODELS["huggingface"]) == 8
# İyi — ilişkiyi test eder
assert "gemini" in _PROVIDER_MODELS
assert len(_PROVIDER_MODELS["gemini"]) >= 1
```

---

## Önemli Politikalar

### Prompt Önbellekleme Kırılmamalıdır

Hermes, bir konuşma boyunca önbelleklemenin geçerli kalmasını sağlar. **Şunları yapacak değişiklikleri uygulamayın:**
- Konuşma sırasında geçmiş bağlamı değiştirmek
- Konuşma sırasında araç setlerini değiştirmek
- Konuşma sırasında sistem promptlarını yeniden yüklemek

Önbellek kırılması maliyetleri önemli ölçüde artırır. Yalnızca bağlam sıkıştırma sırasında bağlam değiştirilir.

### Profiller: Çoklu Örnek Desteği

Hermes, her biri kendi `HERMES_HOME` dizinine sahip izole örnekler (profil) destekler. Tüm yollar için `get_hermes_home()` kullanın, asla `~/.hermes` sabit kodlamayın. Kullanıcı mesajları için `display_hermes_home()` kullanın. Testler `~/.hermes/` dizinine yazmamalıdır.

---

## Pull Request Süreci

### Dal adlandırma

```
fix/aciklama         # Hata düzeltmeleri
feat/aciklama        # Yeni özellikler
docs/aciklama        # Dokümantasyon
test/aciklama        # Testler
refactor/aciklama    # Kod yeniden yapılandırması
```

### Göndermeden önce

1. **Testleri çalıştırın**: `scripts/run_tests.sh` (önerilir; CI ile aynı) veya venv etkinken `pytest tests/ -v`
2. **Manuel test edin**: `hermes` çalıştırın ve değiştirdiğiniz kod yolunu deneyin
3. **Çapraz platform etkisini kontrol edin**: Dosya G/Ç, süreç yönetimi veya terminal koduna dokunduysanız macOS, Linux ve WSL2'yi göz önünde bulundurun
4. **PR'leri odaklı tutun**: PR başına tek mantıksal değişiklik. Hata düzeltmesini yeniden yapılandırma veya yeni özellikle karıştırmayın.

### PR açıklaması

Şunları ekleyin:
- **Ne** değişti ve **neden**
- **Nasıl test edileceği** (hatalar için yeniden üretme adımları, özellikler için kullanım örnekleri)
- **Hangi platformlarda** test ettiğiniz
- İlgili issue'lara referans verin

### Commit mesajları

[Conventional Commits](https://www.conventionalcommits.org/) kullanıyoruz:

```
<tür>(<kapsam>): <açıklama>
```

| Tür | Kullanım |
|-----|----------|
| `fix` | Hata düzeltmeleri |
| `feat` | Yeni özellikler |
| `docs` | Dokümantasyon |
| `test` | Testler |
| `refactor` | Kod yeniden yapılandırması (davranış değişikliği yok) |
| `chore` | Derleme, CI, bağımlılık güncellemeleri |

Kapsamlar: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`, vb.

Örnekler:
```
fix(cli): save_config_value'da model string olduğunda çökmeyi önle
feat(gateway): WhatsApp çoklu kullanıcı oturum izolasyonu ekle
fix(security): sudo şifre iletiminde shell injection'ı önle
test(tools): file_operations için birim testleri ekle
```

---

## Issue Bildirme

- [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues) kullanın
- Şunları ekleyin: İşletim sistemi, Python sürümü, Hermes sürümü (`hermes version`), tam hata izi
- Yeniden üretme adımlarını ekleyin
- Kopya oluşturmadan önce mevcut issue'ları kontrol edin
- Güvenlik açıkları için lütfen özel olarak bildirin

---

## Topluluk

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch) — sorular, projeleri sergileme ve yetenek paylaşımı için
- **GitHub Discussions**: Tasarım önerileri ve mimari tartışmaları için
- **Skills Hub**: Uzmanlaşmış yetenekleri bir kayda yükleyin ve toplulukla paylaşın

---

## Lisans

Katkıda bulunarak, katkılarınızın [MIT Lisansı](LICENSE) altında lisanslanacağını kabul etmiş olursunuz.
