# Hermes Agent'a Katkıda Bulunma

Hermes Agent'a katkıda bulunduğunuz için teşekkürler! Bu kılavuz ihtiyacınız olan her şeyi kapsar: geliştirme ortamınızı kurmak, mimariyi anlamak, ne inşa edeceğinize karar vermek ve PR'inizin birleştirilmesini sağlamak.

---

## Katkı Öncelikleri

Katkıları şu sırayla değerlendiririz:

1. **Hata düzeltmeleri** — çökmeler, yanlış davranış, veri kaybı. Her zaman en yüksek öncelik.
2. **Platformlar arası uyumluluk** — macOS, farklı Linux dağıtımları ve Windows'ta WSL2. Hermes'in her yerde çalışmasını istiyoruz.
3. **Güvenlik sertleştirme** — shell enjeksiyonu, prompt enjeksiyonu, yol geçişi, ayrıcalık yükseltme. [Güvenlik Hususları](#güvenlik-hususları) bölümüne bakın.
4. **Performans ve sağlamlık** — yeniden deneme mantığı, hata yönetimi, zarif degrade.
5. **Yeni beceriler** — ancak yalnızca geniş çapta yararlı olanlar. [Beceri mi yoksa Araç mı olmalı?](#beceri-mi-yoksa-arac-mi-olmalı) bölümüne bakın.
6. **Yeni araçlar** — nadiren gerekir. Yeteneklerin çoğu beceri olmalıdır. Aşağıya bakın.
7. **Dokümantasyon** — düzeltmeler, açıklamalar, yeni örnekler.

---

## Başlamadan Önce: Önce Arayın

Kod yazmadan önce hızlı bir arama zamanınızı kurtarır ve PR sırasını temiz tutar — yinelenenler burada yaygındır, bu yüzden başta bir dakika harcamaya değer.

- **Hem açık hem birleştirilmiş PR'leri ve issue'ları arayın** — konunuz veya hata belirtiniz için. PR şablonundaki yinelenen kontrolü, işi tamamladıktan sonra inceleme sırasında çalışır:
  ```bash
  gh search issues --repo NousResearch/hermes-agent "<terimleriniz>"
  gh search prs --repo NousResearch/hermes-agent --state all "<terimleriniz>"
  ```
  veya web arayüzünü kullanın: [issue'lar](https://github.com/NousResearch/hermes-agent/issues?q=) · [PR'ler (tüm durumlar)](https://github.com/NousResearch/hermes-agent/pulls?q=is%3Apr).
- **Issue takipçisi koddan geri kalabilir.** İstenen özelliklerin çoğu zaten ağaç içinde uygulanmıştır, bu yüzden önermeden önce kaynakta da arayın (`search_files` veya editörünüzün grep'i).
- **Açık bir PR zaten adresliyorsa**, rakip bir yinelenen açmak yerine o PR'ı incelemeyi veya geliştirmeyi düşünün.
- **Daha büyük işler için**, üzerinde çalıştığınızı belirtmek için issue'ya yorum yapın, böylece başkaları aynı şeyi başlatmaz.

---

## Beceri mi yoksa Araç mı Olmalı?

Bu, yeni katkıda bulunanlar için en sık sorulan sorudur. Cevap neredeyse her zaman **beceri**dir.

### Beceri yapın, şu durumlarda:

- Yetenek talimatlar + shell komutları + mevcut araçlar olarak ifade edilebiliyorsa
- Bir dış CLI veya API'yi sarmalar ki ajan `terminal` veya `web_extract` üzerinden çağırabilir
- Özel Python entegrasyonu veya ajana entegre API anahtarı yönetimi gerektirmiyorsa
- Örnekler: arXiv arama, git iş akışları, Docker yönetimi, PDF işleme, CLI araçlarıyla e-posta

### Araç yapın, şu durumlarda:

- API anahtarları, kimlik doğrulama akışları veya ajan harness tarafından yönetilen çok bileşenli yapılandırma ile uçtan uca entegrasyon gerektiriyorsa
- Her seferinde kesin olarak çalışması gereken özel işleme mantığı gerektiriyorsa (LLM yorumlamasından "elinden gelenin en iyisi" değil)
- Terminalden geçemeyen ikili veri, akış veya gerçek zamanlı olayları işliyorsa
- Örnekler: tarayıcı otomasyonu (Browserbase oturum yönetimi), TTS (ses kodlama + platform teslimatı), görüntü analizi (base64 görüntü işleme)

### Beceri paketlenmeli mi?

Paketlenmiş beceriler (`skills/` içinde) her Hermes kurulumıyla birlikte gelir. **Çoğu kullanıcı için geniş çapta yararlı** olmalıdır:

- Doküman işleme, web araştırma, yaygın geliştirme iş akışları, sistem yönetimi
- Geniş bir kişi yelpazesi tarafından düzenli olarak kullanılır

Beceriniz resmi ve yararlı ancak evrensel olarak gerekli değilse (ör. ücretli bir hizmet entegrasyonu, ağır bir bağımlılık), **`optional-skills/`** içine koyun — repo ile birlikte gelir ancak varsayılan olarak etkinleştirilmez. Kullanıcılar `hermes skills browse` ile keşfedebilir ("resmi" etiketiyle) ve `hermes skills install` ile kurabilir (üçüncü taraf uyarısı yok, yerleşik güven).

Beceriniz özelleştirilmiş, topluluk katkılı veya niş ise, bir **Skills Hub** için daha uygundur — bir beceri kaydına yükleyin ve [Nous Research Discord](https://discord.gg/NousResearch)'da paylaşın. Kullanıcılar `hermes skills install` ile kurabilir.

---

## Bellek Sağlayıcıları: Bağımsız Eklenti Olarak Yayınlayın

**Bu repoya artık yeni bellek sağlayıcıları kabul etmiyoruz.** `plugins/memory/` altındaki yerleşik sağlayıcı seti (byterover, holographic, retaindb) kapalıdır ve eski ağaç içi sağlayıcılar hindsight, honcho, supermemory, mem0 ve openviking artık eklenti kataloğundan yayınlanıyor. Yeni bir bellek arka ucu eklemek istiyorsanız, kullanıcıların `~/.hermes/plugins/` içine kurduğu (veya pip entry point ile) bir **bağımsız eklenti reposu** olarak yayınlayın.

Bağımsız bellek eklentileri:

- Aynı `MemoryProvider` ABC'sini uygular (`agent/memory_provider.py`) — `sync_turn`, `prefetch`, `shutdown` ve isteğe bağlı olarak kurulum sihirbazı entegrasyonu için `post_setup(hermes_home, config)`
- Aynı keşif sistemini kullanır — `discover_memory_providers()` bunları kullanıcı/proje eklenti dizinlerinden ve pip entry point'lerinden alır
- `hermes memory setup` ile `post_setup()` üzerinden entegre olur — çekirdek koda dokunmaya gerek yok
- Bir `cli.py` dosyasında `register_cli(subparser)` üzerinden kendi CLI alt komutlarını kaydedebilir
- Ağaç içi sağlayıcılarla aynı yaşam döngüsü kancalarını ve yapılandırma tesisatını alır

`plugins/memory/` altında yeni bir dizin ekleyen PR'ler, sağlayıcıyı kendi reposu olarak yayınlayacak bir işaretçiyle kapatılır. Mevcut ağaç içi sağlayıcılar kalır; bunlara hata düzeltmeleri hoş karşılanır.

Bu bir kalite çubuğu değildir — bir bağlaşım ve bakım kararıdır. Bellek sağlayıcıları en yaygın eklenti türüdür ve hepsinin bu ağaçta yaşaması gerekmez.

---

## Geliştirme Kurulumu

### Ön koşullar

| Gereksinim | Notlar |
|------------|--------|
| **Git** | `git-lfs` uzantısı kurulu |
| **Python 3.14** | Proje `>=3.14,<3.15` gerektirir; PM sabitlenmiş yorumlayıcıyı sağlar |
| **Node.js** | PM sabitlemesini kullanın veya kök `package.json` tarafından kabul edilen bir sürüm: `^22.22.0`, `^24.11.0` veya `>=26.0.0` |

### PM geliştirici ortamı

Hazırlık, aktivasyon, günlük komutlar, bağımlılık değişiklikleri ve test ortamları için [PM geliştirici iş akışını](website/docs/reference/package-management.md#developer-workflow) kullanın. Kurulumdan önce geliştirme `HERMES_HOME`'unuzu seçin, böylece deneysel kod üretim verilerine geçmez.

Her yeni kabukta depo kökünden etkinleştirin. Aktivasyon, kurulumun yalnızca çalışma zamanı yolunu çalıştırır, bu yüzden taze bir checkout sağlar ve eski bağımlılıkları senkronize eder.

Bash:

```bash
source ./activate
hermes --version
```

fish:

```fish
source ./activate.fish
hermes --version
```

PowerShell:

```powershell
. .\activate.ps1
hermes --version
```

Bu checkout için `hermes` çalıştırın. Aktivasyon, bunu bu çalışma ağacı için bir işlev olarak tanımlar, bu yüzden global bir `hermes` komutunu veya MSIX takma adını gizler ve çalışma ağacı dışında reddedilir. PM aktivasyonu, araçları ve Python bağımlılıklarını kabuğa eklemeden önce senkronize eder. JS çalışma alanlarını kurmaz veya başlatıcıları ve kabuk yapılandırmasını yeniden yazmaz. `deactivate`, önceki kabuk ortamını geri yükler ve işlevi kaldırır.

Kabuk etkinleştirmeden ortamda tek komut çalıştırmak için `scripts/run-in-hermes-env CMD...` kullanın.

### Bağımsız test ortamı

Önce Python 3.14'ü (`>=3.14,<3.15`) hazırlamak için [PM geliştirici iş akışını](website/docs/reference/package-management.md#developer-workflow) kullanın. Bu komutları o checkout'tan hazırlanmış Python ile çalıştırın. Aynı geliştirme `HERMES_HOME`'u kullanın. PM, başka bir ortam oluşturmadan önce başlayabilmelidir. Windows'ta, kaynak bağımlılıklarını oluşturmadan önce mimariniz için yerel C++ yapı ortamını başlatın.

Testler ve editör araçları için bağımsız bir yorumlayıcı oluşturun:

```bash
python -m pm.build_env --source . --out .venv --group dev --group test
```

PM, işlenmiş lock'tan oluşturur ve yeni yorumlayıcıyı döndürmeden önce bağımlılık tutarlılığını kontrol eder. `test` grubu yerel başlatıcı test bağımlılıklarını içerir ve uygulama çalışma zamanına girmez. Testler başka bir tanımlı özellik gerektiriyorsa, `--extra`'sını ekleyin.

Çıktı yolu var olmamalıdır — boş bir dizin veya sembolik bağlantı bile. Bir bağımlılık değişikliğinden sonra yeniden oluşturmak için süreçlerini durdurun ve yalnızca o elden çıkarılabilir ortamı bilinçli olarak kaldırın. PM mevcut bir hedefi silmez. PM tarafından oluşturulmuş bir ortamı değiştirmek için ham pip veya uv komutları çalıştırmayın.

Test ortamını checkout dışında tutmak için `.venv` yerine taze bir mutlak yol kullanın. `HERMES_PYTHON`'ı o ortamın yorumlayıcısına ayarlayın:

- POSIX: `export HERMES_PYTHON="/mutlak/yol/hermes-dev/bin/python"`
- PowerShell: `$env:HERMES_PYTHON = 'C:\mutlak\yol\hermes-dev\Scripts\python.exe'`

Kanonik koşucu, depo `.venv`'ini otomatik keşfeder. `PYTHONPATH`'i temizler, bu yüzden pytest yorumlayıcının kendi ortamında kurulu olmalıdır. Bu test ortamı, PM'nin uygulama seçimini veya araç deposunu değiştirmez. Paketlenmiş bir uygulamayı ona yönlendirmeyin veya bir MSIX yüküne kurmayın.

İzole bir geliştirme örneği için, kaynak komutunu başlatmadan önce elden çıkarılabilir bir `HERMES_HOME` seçin. Üretim kimlik bilgilerini checkout'a kopyalamak yerine yapılandırmak için `hermes setup` kullanın.

`pyproject.toml`'u değiştirirseniz, lock'u yeniden oluşturun, aktivasyonu yeniden yükleyin ve `pyproject.toml`'u `uv.lock` ile birlikte doğrulayın:

```bash
hermes pm lock
source ./activate
```

JavaScript için, ilgili çalışma alanında `npm ci` çalıştırın. Tam gereksinimler için [CONTRIBUTING.md](CONTRIBUTING.md)'ye bakın.

---

## Proje Yapısı

```
hermes-agent/
├── run_agent.py              # AIAgent sınıfı — çekirdek konuşma döngüsü, araç dağıtımı, oturum kalıcılığı
├── cli.py                    # HermesCLI sınıfı — etkileşimli TUI, prompt_toolkit entegrasyonu
├── model_tools.py             # Araç orkestrasyonu (tools/registry.py üzerine ince katman)
├── toolsets.py                # Araç grupları ve ön ayarları (hermes-cli, hermes-telegram vb.)
├── hermes_state.py            # FTS5 tam metin aramalı SQLite oturum veritabanı, oturum başlıkları
├── batch_runner.py            # Yörünge üretimi için paralel toplu işleme
│
├── agent/                     # Ajan iç işleri (ayrıştırılmış modüller)
│   ├── prompt_builder.py         # Sistem promptu montajı (kimlik, beceriler, bağlam dosyaları, bellek)
│   ├── context_compressor.py     # Bağlam sınırına yaklaşınca otomatik özetleme
│   ├── auxiliary_client.py       # Yardımcı OpenAI istemcilerini çözer (özetleme, görüntü)
│   ├── display.py                # KawaiiSpinner, araç ilerleme biçimlendirme
│   ├── model_metadata.py         # Model bağlam uzunlukları, token tahmini
│   └── trajectory.py             # Yörünge kaydetme yardımcıları
│
├── hermes_cli/                # CLI komut uygulamaları
│   ├── main.py                   # Giriş noktası, argüman analizi, komut dağıtımı
│   ├── config.py                 # Yapılandırma yönetimi, geçiş, ortam değişkeni tanımları
│   ├── setup.py                  # Etkileşimli kurulum sihirbazı
│   ├── auth.py                   # Sağlayıcı çözümü, OAuth, Nous Portal
│   ├── models.py                 # OpenRouter model seçim listeleri
│   ├── banner.py                 # Karşılama afişi, ASCII sanat
│   ├── commands.py               # Merkezi slash komut kaydı (CommandDef), otomatik tamamlama, gateway yardımcıları
│   ├── callbacks.py              # Etkileşimli callback'ler (açıklığa kavuşturma, sudo, onay)
│   ├── doctor.py                 # Teşhisler
│   ├── skills_hub.py             # Skills Hub CLI + /skills slash komutu
│   └── skin_engine.py            # Skin/tema motoru — veri tabanlı CLI görsel kişiselleştirme
│
├── tools/                     # Araç uygulamaları (otomatik kayıtlı)
│   ├── registry.py               # Merkezi araç kaydı (şemalar, işleyiciler, dağıtım)
│   ├── approval.py               # Tehlikeli komut tespiti + oturum onayı
│   ├── terminal_tool.py          # Terminal orkestrasyonu (sudo, ortam yaşam döngüsü, arka uçlar)
│   ├── file_operations.py        # read_file, write_file, arama, patch vb.
│   ├── web_tools.py              # web_search, web_extract (Paralel/Firecrawl + Gemini özetleme)
│   ├── vision_tools.py           # Multimodal modellerle görüntü analizi
│   ├── delegate_tool.py          # Alt ajan başlatma ve paralel görev yürütme
│   ├── code_execution_tool.py    # RPC üzerinden araç erişimli sandboxed Python
│   ├── session_search_tool.py    # FTS5 + sabitleme pencereleriyle geçmiş konuşmalarda arama
│   ├── cronjob_tools.py          # Zamanlanmış görev yönetimi
│   ├── skill_tools.py            # Beceri arama, yükleme ve yönetim
│   └── environments/             # Terminal yürütme arka uçları
│       ├── base.py                   # BaseEnvironment ABC
│       ├── local.py, docker.py, ssh.py, singularity.py, modal.py, daytona.py
│
├── gateway/                  # Mesajlaşma gateway'i
│   ├── run.py                    # GatewayRunner — platform yaşam döngüsü, mesaj yönlendirme, cron
│   ├── config.py                 # Platform yapılandırma çözümü
│   ├── session.py                # Oturum deposu, bağlam promptları, sıfırlama politikaları
│   └── platforms/                # Platform adaptörleri
│       ├── telegram.py, discord_adapter.py, slack.py, whatsapp.py
│
├── scripts/                  # Kurulum ve köprü betikleri
│   ├── install.sh                # Linux/macOS kurulumu
│   ├── install.ps1               # Windows PowerShell kurulumu
│   └── whatsapp-bridge/          # WhatsApp Node.js köprüsü (Baileys)
│
├── skills/                   # Paketlenmiş beceriler (kurulumda ~/.hermes/skills/'a kopyalanır)
├── optional-skills/          # Resmi isteğe bağlı beceriler (hub üzerinden keşfedilir, varsayılan kapalı)
├── tests/                    # Test paketi
├── website/                  # Dokümantasyon sitesi (hermes-agent.nousresearch.com)
│
├── cli-config.yaml.example   # Örnek yapılandırma (~/.hermes/config.yaml'a kopyalanır)
└── AGENTS.md                 # AI kodlama asistanları için geliştirme kılavuzu
```

### Kullanıcı yapılandırması (`~/.hermes/` içinde saklanır)

| Yol | Amaç |
|-----|------|
| `~/.hermes/config.yaml` | Yapılandırma (model, terminal, araç setleri, sıkıştırma vb.) |
| `~/.hermes/.env` | API anahtarları ve gizli veriler |
| `~/.hermes/auth.json` | OAuth kimlik bilgileri (Nous Portal) |
| `~/.hermes/skills/` | Tüm etkin beceriler (paketlenmiş + hub'dan kurulan + ajan tarafından oluşturulan) |
| `~/.hermes/memories/` | Kalıcı bellek (MEMORY.md, USER.md) |
| `~/.hermes/state.db` | SQLite oturum veritabanı |
| `~/.hermes/sessions/` | Gateway yönlendirme indeksi (`sessions.json`), istek kırıntıları, `*.jsonl` gateway transkripsiyonları ve `/save` ile açık dışa aktarmalar. Otomatik JSON anlık görüntüleri artık yazılmıyor; mevcut dosyalar korunur ve state.db kanoniktir. |
| `~/.hermes/cron/` | Zamanlanmış iş verileri |
| `~/.hermes/whatsapp/session/` | WhatsApp köprü kimlik bilgileri |

---

## Mimariye Genel Bakış

### Çekirdek Döngü

```
Kullanıcı mesajı → AIAgent._run_agent_loop()
  ├── Sistem promptu oluştur (prompt_builder.py)
  ├── API kwargs oluştur (model, mesajlar, araçlar, akıl yürütme yapılandırması)
  ├── LLM'i çağır (OpenAI uyumlu API)
  ├── Yanıtta tool_calls varsa:
  │     ├── Her aracı kayıt dağıtımı üzerinden yürüt
  │     ├── Araç sonuçlarını konuşmaya ekle
  │     └── LLM çağrısına geri dön
  ├── Metin yanıtı varsa:
  │     ├── Oturumu DB'ye kaydet
  │     └── final_response döndür
  └── Token sınırına yaklaşırsa bağlam sıkıştırma
```

### Temel Tasarım Desenleri

- **Otomatik kayıtlı araçlar**: Her araç dosyası import zamanında `registry.register()` çağırır. `model_tools.py`, tüm araç modüllerini import ederek keşfi etkinleştirir.
- **Araç setlerinde gruplama**: Araçlar, platform bazında etkinleştirilebilir/devre dışı bırakılabilen araç setleri (`web`, `terminal`, `file`, `browser` vb.) içinde gruplanır.
- **Oturum kalıcılığı**: Tüm konuşmalar SQLite'te (`hermes_state.py`) tam metin araması ve benzersiz oturum başlıklarıyla saklanır.
- **Geçici enjeksiyon**: Sistem promptları ve dolgu mesajları API çağrısı zamanında enjekte edilir, hiçbir zaman veritabanında veya günlüklerde kalıcı olmaz.
- **Sağlayıcı soyutlaması**: Ajan, OpenAI uyumlu herhangi bir API ile çalışır. Sağlayıcı çözümü başlatma zamanında olur.
- **Sağlayıcı yönlendirme**: OpenRouter kullanıldığında, `config.yaml`'da `provider_routing` sağlayıcı seçimini kontrol eder.

---

## Kod Stili

- **PEP 8** pratik istisnalarla (katı satır uzunluğu dayatmıyoruz)
- **Yorumlar**: Yalnızca bariz olmayan niyeti, ödünleşimleri veya API tuhaflıklarını açıklarken. Kodun ne yaptığını anlatmayın
- **Hata yönetimi**: Belirli istisnaları yakalayın. `logger.warning()`/`logger.error()` ile kaydedin — beklenmeyen hatalar için `exc_info=True` kullanın
- **Çok platformlu**: Unix asla varsaymayın. [Çoklu Platform Uyumluluğu](#coklu-platform-uyumlulugu) bölümüne bakın

---

## Yeni Araç Ekleme

Bir araç yazmadan önce kendinize sorun: [Bunun yerine bir beceri olmalı mı?](#beceri-mi-yoksa-arac-mi-olmalı)

Araçlar merkezi kayda otomatik kaydedilir. Her araç dosyası şemasını, işleyicisini ve kaydını birlikte konumlandırır:

```python
"""my_tool — Bu aracın ne yaptığı hakkında kısa açıklama."""

import json
from tools.registry import registry


def my_tool(param1: str, param2: int = 10, **kwargs) -> str:
    """İşleyici. Bir dize sonucu döndürür (genellikle JSON)."""
    result = do_work(param1, param2)
    return json.dumps(result)


MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Bu araç ne yapar ve ajan ne zaman kullanmalı.",
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
    """Bu aracın bağımlılıkları mevcutsa True döndürür."""
    return True


registry.register(
    name="my_tool",
    toolset="my_toolset",
    schema=MY_TOOL_SCHEMA,
    handler=lambda args, **kw: my_tool(**args, **kw),
    check_fn=_check_requirements,
)
```

**Araç setine bağlama (gerekli):** Yerleşik araçlar otomatik keşfedilir — üst düzey bir
`registry.register(...)` çağrısı içeren her `tools/*.py` dosyası, `model_tools`
yüklendiğinde `tools/registry.py`'deki `discover_builtin_tools()` tarafından import edilir.
`model_tools.py`'de bakımı gereken manuel bir import listesi **yok**.

Yine de araç adını `toolsets.py`'deki uygun listeye eklemeniz gerekir
(ör. `_HERMES_CORE_TOOLS` veya özel bir araç seti); aksi takdirde araç kaydedilir
ama ajana asla gösterilmez.

Profil farkında yollar ve eklenti vs. çekirdek konusunda rehberlik için `AGENTS.md`'ye
(**Adding New Tools** bölümü) bakın.

---

## Beceri Ekleme

Paketlenmiş beceriler `skills/` içinde kategoriye göre düzenlenir. Resmi isteğe bağlı beceriler `optional-skills/` içinde aynı yapıyı kullanır:

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Gerekli: ana talimatlar
│       └── scripts/              # İsteğe bağlı: yardımcı betikler
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

### SKILL.md Formatı

```markdown
---
name: my-skill
description: Kısa açıklama (beceri arama sonuçlarında gösterilir)
version: 1.0.0
author: Adınız
license: MIT
platforms: [macos, linux]          # İsteğe bağlı — belirli işletim sistemi platformlarıyla sınırla
required_environment_variables:    # İsteğe bağlı — güvenli yapılandırma meta verileri
  - name: MY_API_KEY
    prompt: API Anahtarı
    help: Nereden alınır
    required_for: tam işlevsellik
prerequisites:                     # İsteğe bağlı devralınan çalışma zamanı gereksinimleri
  env_vars: [MY_API_KEY]
  commands: [curl, jq]
metadata:
  hermes:
    tags: [Kategori, Alt Kategori, Anahtar Kelimeler]
    related_skills: [other-skill-name]
    fallback_for_toolsets: [web]
    requires_toolsets: [terminal]
---

# Beceri Başlığı

Kısa giriş.

## Ne Zaman Kullanılır
Tetikleme koşulları — ajan bu beceriyi ne zaman yüklemeli?

## Hızlı Başvuru
Yaygın komutlar veya API çağrıları tablosu.

## Prosedür
Ajanın izlediği adım adım talimatlar.

## Bilinen Sorunlar
Bilinen başarısızlık modları ve nasıl ele alınacağı.

## Doğrulama
Ajan, işin başarılı olduğunu nasıl doğrular.
```

### Beceri yazma standartları (ZORUNLU)

Tüm yeni veya modernize edilmiş beceriler — paketlenmiş, isteğe bağlı veya katkıda bulunulan — birleştirmeden önce bu standartları karşılamalıdır:

1. **`description` ≤ 60 karakter, bir cümle, noktayla bitmeli.** Uzun açıklamalar beceri listeleme UI'sını doldurur. Yeteneği belirtin, uygulamayı değil. Pazarlama kelimeleri yok ("güçlü", "kapsamlı", "kusursuz", "gelişmiş").

2. **SKILL.md gövdesinde referans verilen araçlar, yerel Hermes araçları veya becerinin açıkça beklediği MCP sunucuları olmalıdır.** Araç adlarını ters tırnak içinde kullanın: `` `terminal` ``, `` `web_extract` ``, `` `web_search` ``, `` `read_file` ``, `` `write_file` `` vb.

3. **`platforms:` alanı, betiğin gerçek importlarına karşı denetlenir.** Yalnızca POSIX temel öğeleri kullanan beceriler, desteklenen platformlarını belirtmelidir.

4. **`author`, önce insan katkıda bulunana kredi verir.**

5. **SKILL.md gövdesi modern bölüm sırasını kullanır:** başlık, 2-3 cümlelik giriş, ardından: `## Ne Zaman Kullanılır`, `## Ön Koşullar`, `## Nasıl Çalıştırılır`, `## Hızlı Başvuru`, `## Prosedür`, `## Bilinen Sorunlar`, `## Doğrulama`.

6. **Betikler `scripts/` içinde, referanslar `references/` içinde, şablonlar `templates/` içinde.**

7. **Testler `tests/skills/test_<skill>_skill.py` içinde** ve yalnızca stdlib + pytest + `unittest.mock` kullanır. Canlı ağ çağrısı yok.

8. **`.env.example` eklemeleri açıkça sınırlandırılmış bir blok içinde olmalı.**

---

## Skin / Tema Ekleme

Hermes, veri tabanlı bir skin sistemi kullanır — yeni bir skin eklemek için kod değişikliği gerekmez.

**Seçenek A: Kullanıcı skini (YAML dosyası)**

`~/.hermes/skins/<isim>.yaml` oluşturun:

```yaml
name: temam
description: Tema hakkında kısa açıklama

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
  welcome: "Karşılama mesajı"
  response_label: " ⚔ Ajan "
  prompt_symbol: "⚔"

tool_prefix: "╎"
```

Tüm alanlar isteğe bağlıdır — eksik değerler varsayılan skinden devralınır.

**Seçenek B: Yerleşik skin**

`hermes_cli/skin_engine.py`'deki `_BUILTIN_SKINS` dict'ine ekleyin. Yukarıdaki şemayı kullanın ama Python dict'i olarak.

**Etkinleştir:**
- CLI: `/skin temam` veya `config.yaml`'da `display.skin: temam` ayarlayın

---

## Çoklu Platform Uyumluluğu

Hermes Linux, macOS ve native Windows'ta (artı WSL2) çalışır. işletim sistemine
dokunan kod yazarken, *herhangi bir* platformun kod yolunuza ulaşabileceğini varsayın.

> **PR'den önce:** diff'inizdeki yaygın Windows tuzağılarını tespitmek için
> `scripts/check-windows-footguns.py` çalıştırın. grep tabanlıdır ve ucuzdur;
> CI da her PR'da çalıştırır.

### Kritik kurallar

1. **Canlılık kontrolü için asla `os.kill(pid, 0)` çağırmayın.** Windows'ta **etkisiz bir işlem değildir**. Bunun yerine `psutil.pid_exists(pid)` kullanın.

2. **Shell yapmadan önce `shutil.which()` kullanın — Windows'un Linux'un sahip olduğu araçlara sahip olduğunu varsaymayın.** `ps`, `kill`, `grep`, `awk` vb. Windows'ta basitçe yoktur.

3. **`termios` ve `fcntl` yalnızca Unix'te.** Her zaman hem `ImportError` hem `NotImplementedError` yakalayın.

4. **Dosya kodlaması.** Windows `.env` dosyalarını `cp1252` olarak kaydedebilir. Her zaman kodlama hatalarını yönetin.

5. **Süreç yönetimi.** `os.setsid()`, `os.killpg()`, `os.fork()`, `os.getuid()` ve POSIX sinyal yönetimi Windows'ta farklıdır.

6. **Windows'ta var olmayan sinyaller:** `SIGALRM`, `SIGCHLD`, `SIGHUP`, `SIGUSR1`, `SIGUSR2` vb.

7. **Yol ayırıcıları.** `/` ile dize birleştirme yerine `pathlib.Path` kullanın.

8. **Sembolik bağlantılar Windows'ta yükseltilmiş ayrıcalık gerektirir** (Geliştirici Modu açık değilse).

9. **POSIX dosya modları (0o600, 0o644 vb.) NTFS'te varsayılan olarak UYGULANMAZ.**

10. **Windows'taki arka plan daemon'ları `pythonw.exe` gerektirir, `python.exe` değil.**

---

## Güvenlik Hususları

Hermes'in terminale erişimi var. Güvenlik önemli.

### Mevcut korumalar

| Katman | Uygulama |
|--------|----------|
| **Sudo parola borulama** | Shell enjeksiyonunu önlemek için `shlex.quote()` kullanır |
| **Tehlikeli komut tespiti** | `tools/approval.py`'de regex desenleri + kullanıcı onay akışı |
| **Cron prompt enjeksiyonu** | `tools/cronjob_tools.py`'de tarayıcı, talimat geçersiz kılma desenlerini engeller |
| **Yazma reddetme listesi** | Sembolik bağ bypassını önlemek için `os.path.realpath()` ile çözümlenen korumalı yollar |
| **Skills Guard** | Hub'dan kurulan beceriler için güvenlik tarayıcısı (`tools/skills_guard.py`) |
| **Kod yürütme sandbox'ı** | `execute_code` alt süreci, ortamdan API anahtarları çıkarılarak çalışır |
| **Konteyner sertleştirme** | Docker: tüm yetenekler kaldırıldı, ayrıcalık yükseltme yok, PID sınırları, sınırlı boyutlu tmpfs |

### Güvenlik açısından hassas koda katkıda bulunurken

- Kullanıcı girdisini shell komutlarına enterpole ederken **her zaman `shlex.quote()` kullanın**
- Yol tabanlı erişim kontrolü kontrollerinden önce sembolik bağlantıları `os.path.realpath()` ile çözün
- **Gizli verileri kaydetmeyin.** API anahtarları, token'lar ve parolalar asla günlük çıktısında görünmemeli
- Tek bir hatanın ajan döngüsünü engellememesi için araç yürütme etrafında **geniş istisna yakalama**
- Dosya yollarına, süreç yönetimine veya shell komutlarına dokunan değişikliklerde **tüm platformlarda test edin**

### Bağımlılık Sabitleme Politikası (tedarik zinciri sertleştirme)

Mart 2026'daki [litellm tedarik zinciri ihlali](https://github.com/BerriAI/litellm/issues/24512) ve Mayıs 2026'daki [Mini Shai-Hulud solucan kampanyası](https://socket.dev/blog/tanstack-npm-packages-compromised-mini-shai-hulud-supply-chain-attack)'dan sonra, tüm bağımlılıklar bu kuralları izlemelidir:

| Kaynak Türü | Gerekli İşlem | Gerekçe |
|---|---|---|
| **PyPI paketi** | `>=zemin,<sonraki_majör` | PyPI sürümleri yayınlandıktan sonra değişmezdir, ancak aralığınıza yeni sürümler itilebilir. |
| **Git URL'si** | Tam commit SHA | Dallar ve etiketler değiştirilebilir refs; SHA içerik tarafından adreslenir. |
| **GitHub Actions** | Tam commit SHA + sürüm yorumu | Action etiketleri değiştirilebilir refs. `uses: owner/action@<sha>  # vX.Y.Z` olarak sabitleyin |
| **Yalnızca CI pip kurulumları** | `==tam` | Hermetik CI derlemeleri; değişiklik kabul edilebilir. |

**Bir PR'deki her yeni PyPI bağımlılığı bir üst sınıra `<sonraki_majör` sahip olmalıdır.** Üst sınır olmadan `>=X.Y.Z` belirtimi ekleyen PR'ler reddedilir.

---

## Pull Request Süreci

### Dal adlandırma

```
fix/aciklama         # Hata düzeltmeleri
feat/aciklama        # Yeni özellikler
docs/aciklama        # Dokümantasyon
test/aciklama        # Testler
refactor/aciklama    # Kod yeniden yapılandırma
```

### Göndermeden önce

1. **Testleri çalıştırın**: `scripts/run_tests.sh` (önerilir; CI ile aynı) veya proje venv'i etkinleştirilmişken `pytest tests/ -v`
2. **Manuel test edin**: `hermes` çalıştırın ve değiştirdiğiniz kod yolunu egzersiz edin
3. **Çoklu platform etkisini doğrulayın**: Dosya G/Ç, süreç yönetimi veya terminal yönetimine dokunuyorsanız macOS, Linux ve WSL2'yi düşünün
4. **PR'leri odaklı tutun**: PR başına bir mantıksal değişiklik. Bir hata düzeltmesini yeniden yapılandırma ile yeni özellikle karıştırmayın.

### PR açıklaması

Şunları ekleyin:
- **Ne** değişti ve **neden**
- **Nasıl test edilir** (hatalar için yeniden üretme adımları, özellikler için kullanım örnekleri)
- **Hangi platformları** test ettiniz
- İlgili issue'ları referanslayın

### Commit mesajları

[Conventional Commits](https://www.conventionalcommits.org/) kullanıyoruz:

```
<tür>(<kapsam>): <açıklama>
```

| Tür | Kullanım |
|------|---------|
| `fix` | Hata düzeltmeleri |
| `feat` | Yeni özellikler |
| `docs` | Dokümantasyon |
| `test` | Testler |
| `refactor` | Kod yeniden yapılandırma (davranış değişikliği yok) |
| `chore` | Derleme, CI, bağımlılık güncellemeleri |

Kapsamlar: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security` vb.

Örnekler:
```
fix(cli): model bir dizeyken save_config_value'da çökmeyi önle
feat(gateway): WhatsApp çok kullanıcılı oturum izolasyonu ekle
fix(security): sudo parola borulamasında shell enjeksiyonunu önle
test(tools): file_operations için birim testleri ekle
```

---

## Issue Bildirme

- [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues) kullanın
- Şunları ekleyin: işletim sistemi, Python sürümü, Hermes sürümü (`hermes --version`), tam hata izleme
- Yeniden üretme adımları ekleyin
- Yinelenen oluşturmadan önce mevcut issue'ları kontrol edin
- Güvenlik açıkları için lütfen özel olarak bildirin

---

## Topluluk

- **Discord**: [discord.gg/NousResearch](https://discord.gg/NousResearch) — sorular, projeleri gösterme ve beceri paylaşma
- **GitHub Discussions**: Tasarım önerileri ve mimari tartışmaları için
- **Skills Hub**: Özelleştirilmiş becerileri bir kayda yükleyin ve toplulukla paylaşın

---

## Lisans

Katkıda bulunarak, katkılarınızın [MIT Lisansı](LICENSE) altında lisanslanacağını kabul edersiniz.
