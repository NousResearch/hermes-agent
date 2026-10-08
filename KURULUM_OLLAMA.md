# LiaAgent + Ollama kurulumu

Bu yönergeler, bu depodaki Ollama tool devam düzeltmesini içeren Hermes sürümünü başka bir bilgisayara kurmak içindir. Kurulum Ollama'yı, seçtiğiniz modeli ve Hermes'i ayrı ayrı hazırlar.

## 1. Bu düzeltilmiş sürümü GitHub'a koyun

Önce GitHub hesabında `LiaAgent` adında boş bir depo oluşturun. Sonra bu depodaki `ollama-tool-followup` branch'ini gönderin. Depo herkese açık olmalı; özel depolar için bilgisayarda GitHub erişimi önceden yapılandırılmalıdır.

Forkunuzu oluşturduktan sonra kendi bilgisayarınızda:

```bash
git remote add myfork https://github.com/Muhammedokbi/LiaAgent.git
git push -u myfork ollama-tool-followup
```

## 2. Yeni bilgisayara Hermes'i kurun

Linux veya macOS terminalinde:

```bash
REPO="https://github.com/Muhammedokbi/LiaAgent.git"
BRANCH="ollama-tool-followup"
curl -fsSL "https://raw.githubusercontent.com/Muhammedokbi/LiaAgent/$BRANCH/scripts/install.sh" -o /tmp/hermes-install.sh
HERMES_REPO_URL="$REPO" bash /tmp/hermes-install.sh --branch "$BRANCH"
```

Bu komut Hermes'in resmi kurulum akışını kullanır, ancak kodu upstream yerine sizin deponuzdaki düzeltilmiş branch'ten alır. Kurulum bittikten sonra yeni bir terminal açın ve `lia --version` ile kontrol edin. Windows'ta Hermes'i WSL2 içinde kurun; Ollama Windows'ta çalışıyorsa WSL'den erişilebilen adresi kullanın.

## 3. Ollama ve modeli kurun

[Ollama'yı indirin](https://ollama.com/download), çalıştığını kontrol edin ve kullandığınız modeli indirin:

```bash
ollama --version
ollama pull qwen3.8:latest
ollama list
```

Modelin disk ve RAM/VRAM ihtiyacı bilgisayara göre değişir. Bu model o makineye sığmıyorsa Ollama'da tool çağrısını destekleyen uygun başka bir model seçebilirsiniz.

## 4. Hermes'i Ollama'ya bağlayın

`lia setup` sihirbazında **Custom Endpoint** seçin ve şunları girin:

- Provider: `custom`
- API mode: `chat_completions`
- Base URL: `http://127.0.0.1:11434/v1`
- API key: boş bırakın
- Model: `qwen3.8:latest`

Gerekirse `~/.hermes/config.yaml` dosyasındaki model bölümü şu şekilde olmalıdır:

```yaml
model:
  default: qwen3.8:latest
  provider: custom
  base_url: http://127.0.0.1:11434/v1
  api_mode: chat_completions
```

Ardından `lia` komutuyla CLI'ı açıp basit bir isteği ve terminal tool çağrısını deneyin. `hermes` komutu uyumluluk için çalışmayı sürdürür. Yerel Ollama için LiaAgent, doğru sunucu türünü algıladığında native `/api/chat` yolunu kullanır; bu yol tool sonucu sonrası devam isteğinde gereken `num_ctx` ayarını destekler.

## Araç ve becerileri aynı tutma

Kod deposunu kurmak Hermes'in araçlarını ve depoda bulunan becerileri getirir. `/help` içindeki toplam sayılar makinedeki etkin toolset'lere, PM araç kurulumlarına ve profilinizdeki becerilere göre değişebilir; başka bilgisayarda yalnızca aynı Git deposunu kurmak `26 tools · 54 skills` sayısını garanti etmez.

Kendi makineleriniz arasında kişisel beceri ve ayarları taşımak için kaynak makinede `/export` ile profil arşivi oluşturup hedefte `/import` kullanın. Profil dışa aktarımı API anahtarlarını çıkarır; ancak kişisel bellek, kullanıcı notları veya oturum verisi içerebilir. Arşivi GitHub'a koymadan önce içeriğini kontrol edin; kişisel arşivleri herkese açık depoya eklemeyin. Düzenli olarak GitHub üzerinden dağıtılacak bir kurulum için profil arşivi yerine Hermes **profile distribution** yöntemi kullanılmalıdır.

Kurulumdan sonra `/help` mevcut komut ve araç ekranını açar. Paylaştığınız `/help` çıktısını ve bununla ilgili sonraki sorunuzu ayrıca not ettim.
