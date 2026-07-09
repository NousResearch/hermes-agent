<div align="center">
<h1 align="center">MoneyPrinterTurbo 💸</h1>

<p align="center">
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/stargazers"><img src="https://img.shields.io/github/stars/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="Stargazers"></a>
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/issues"><img src="https://img.shields.io/github/issues/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="Issues"></a>
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/network/members"><img src="https://img.shields.io/github/forks/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="Forks"></a>
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="License"></a>
</p>

<h3>العربية | <a href="README-en.md">English</a> | <a href="README.md">简体中文</a></h3>

<div align="center">
  <a href="https://trendshift.io/repositories/8731" target="_blank"><img src="https://trendshift.io/api/badge/repositories/8731" alt="harry0703%2FMoneyPrinterTurbo | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

ما عليك سوى تقديم <b>موضوع</b> أو <b>كلمة مفتاحية</b> للفيديو، وسيقوم التطبيق تلقائياً بتوليد نص الفيديو،
ومواد الفيديو، والترجمة، وموسيقى الخلفية، ثم تركيبها في فيديو قصير عالي الدقة.

### واجهة الويب (WebUI)

![](docs/webui-en.jpg)

### واجهة الـ API

![](docs/api.jpg)

</div>

## المميزات 🎯

- [x] بنية **MVC** كاملة، وكود **واضح التنظيم** وسهل الصيانة، يدعم كلاً من `API` و`واجهة الويب`
- [x] يدعم **توليد نص الفيديو بالذكاء الاصطناعي**، إضافةً إلى **النص المخصّص**
- [x] يدعم أحجام **فيديو عالي الدقة** متنوعة
    - [x] عمودي 9:16، `1080x1920`
    - [x] أفقي 16:9، `1920x1080`
- [x] يدعم **توليد الفيديو دفعةً واحدة**، فيمكن إنشاء عدة فيديوهات معاً ثم اختيار الأفضل
- [x] يدعم ضبط **مدة مقاطع الفيديو**، مما يسهّل التحكم في تكرار تبديل المواد
- [x] يدعم نص الفيديو بكل من **الصينية** و**الإنجليزية**
- [x] يدعم **تركيب أصوات متعددة**، مع **معاينة فورية** للنتيجة
- [x] يدعم **توليد الترجمة**، مع إمكانية ضبط `الخط` و`الموضع` و`اللون` و`الحجم`، كما يدعم `تحديد إطار الترجمة`
- [x] يدعم **موسيقى الخلفية**، إما عشوائية أو ملفات موسيقى محدّدة، مع إمكانية ضبط `مستوى صوت موسيقى الخلفية`
- [x] مصادر مواد الفيديو **عالية الدقة** و**خالية من حقوق الملكية**، كما يمكنك استخدام **موادك المحلية** الخاصة
- [x] يدعم التكامل مع نماذج متعددة مثل **OpenAI** و**Moonshot** و**Azure** و**gpt4free** و**one-api** و**Qwen** و**Google Gemini** و**Ollama** و**DeepSeek** و**MiniMax** و**ERNIE** و**Pollinations** و**ModelScope** وغيرها

## عروض فيديو توضيحية 📺

### عمودي 9:16

<table>
<thead>
<tr>
<th align="center"><g-emoji class="g-emoji" alias="arrow_forward">▶️</g-emoji> كيف تضيف المتعة إلى حياتك </th>
<th align="center"><g-emoji class="g-emoji" alias="arrow_forward">▶️</g-emoji> ما معنى الحياة</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><video src="https://github.com/harry0703/MoneyPrinterTurbo/assets/4928832/a84d33d5-27a2-4aba-8fd0-9fb2bd91c6a6"></video></td>
<td align="center"><video src="https://github.com/harry0703/MoneyPrinterTurbo/assets/4928832/112c9564-d52b-4472-99ad-970b75f66476"></video></td>
</tr>
</tbody>
</table>

### أفقي 16:9

<table>
<thead>
<tr>
<th align="center"><g-emoji class="g-emoji" alias="arrow_forward">▶️</g-emoji> ما معنى الحياة</th>
<th align="center"><g-emoji class="g-emoji" alias="arrow_forward">▶️</g-emoji> لماذا تمارس الرياضة</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><video src="https://github.com/harry0703/MoneyPrinterTurbo/assets/4928832/346ebb15-c55f-47a9-a653-114f08bb8073"></video></td>
<td align="center"><video src="https://github.com/harry0703/MoneyPrinterTurbo/assets/4928832/271f2fae-8283-44a0-8aa0-0ed8f9a6fa87"></video></td>
</tr>
</tbody>
</table>

## متطلبات النظام 📦

- المنصّات المُوصى بها: Windows 10+ أو macOS 11+ أو توزيعة Linux رئيسية
- وجود كرت رسومات (GPU) ليس ضرورياً، لكنه مُستحسَن إن أردت نسخاً صوتياً محلياً أسرع، أو معالجة فيديو أسرع، أو توليداً دفعياً أكثر سلاسة

| العنصر | الحد الأدنى | المُوصى به | الأمثل |
| --- | --- | --- | --- |
| المعالج (CPU) | 4 أنوية | 6 إلى 8 أنوية | 8+ أنوية |
| الذاكرة (RAM) | 4 GB | 8 GB | 16+ GB |
| كرت الرسومات (GPU) | غير مطلوب | 4+ GB VRAM | 8+ GB VRAM |

- إذا كنت تعتمد أساساً على نماذج LLM السحابية، وخدمات TTS السحابية، ومصادر المواد عبر الإنترنت، فإن المعالج والذاكرة أهم من كرت الرسومات
- إذا كنت تستخدم `faster-whisper` أو التوليد الدفعي أو المعالجة المحلية الثقيلة، فسيحسّن كرت الرسومات الإنتاجية بشكل ملحوظ

## البدء السريع 🚀

### المسارات المُوصى بها

- مستخدمو Windows: استخدم الحزمة الجاهزة بنقرة واحدة أولاً للتجربة المحلية الأسرع
- مستخدمو MacOS / Linux: استخدم `uv sync --frozen` كمسار الإعداد المحلي الأساسي
- إذا أردت بيئة تشغيل أكثر عزلاً: استخدم النشر عبر Docker

### التشغيل في Google Colab
تريد تجربة MoneyPrinterTurbo دون إعداد بيئة محلية؟ شغّله مباشرةً في Google Colab!

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harry0703/MoneyPrinterTurbo/blob/main/docs/MoneyPrinterTurbo.ipynb)


### Windows

الحزمة القابلة للتنزيل ما زالت بناء `v1.2.6` القديم المُجمّع. بعد التنزيل، شغّل `update.bat` أولاً لتحديثه إلى أحدث كود.

Google Drive (v1.2.6): https://drive.google.com/file/d/1HsbzfT7XunkrCrHw5ncUjFX8XX4zAuUh/view?usp=sharing

بعد التنزيل، يُنصح بالنقر المزدوج على `update.bat` أولاً للتحديث إلى **أحدث كود**، ثم النقر المزدوج على `start.bat` للتشغيل

بعد التشغيل، سيُفتح المتصفح تلقائياً (إن فُتح فارغاً، يُنصح باستخدام **Chrome** أو **Edge**)

### الأنظمة الأخرى

لم تُنشأ حزم تشغيل بنقرة واحدة بعد. راجع قسم **التثبيت والنشر** أدناه. يُنصح باستخدام **docker** للنشر لأنه أكثر سهولة.

## التثبيت والنشر 📥

### المتطلبات المُسبقة

#### ① استنساخ المشروع

```shell
git clone https://github.com/harry0703/MoneyPrinterTurbo.git
```

#### ② تعديل ملف الإعدادات

- انسخ ملف `config.example.toml` وأعد تسميته إلى `config.toml`
- اتبع التعليمات داخل ملف `config.toml` لضبط `pexels_api_keys` و`llm_provider`، وبحسب مزوّد خدمة الـ llm_provider، اضبط مفتاح الـ API المقابل

### النشر عبر Docker 🐳

#### ① تشغيل حاوية Docker

إذا لم تكن قد ثبّت Docker، فثبّته أولاً https://www.docker.com/products/docker-desktop/
إذا كنت تستخدم نظام Windows، فراجع وثائق Microsoft:

1. https://learn.microsoft.com/en-us/windows/wsl/install
2. https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers

```shell
cd MoneyPrinterTurbo
docker-compose up
```

> ملاحظة: أحدث إصدار من docker يثبّت docker compose تلقائياً على هيئة إضافة (plug-in)، ويتغيّر أمر التشغيل إلى `docker compose up`

#### ② الوصول إلى واجهة الويب

افتح متصفحك وزر http://127.0.0.1:8501

#### ③ الوصول إلى واجهة الـ API

افتح متصفحك وزر http://0.0.0.0:8080/docs أو http://0.0.0.0:8080/redoc

### النشر اليدوي 📦

#### ① إنشاء بيئة Python افتراضية

يُنصح باستخدام [uv](https://docs.astral.sh/uv/) لإدارة بيئة Python والاعتماديات، مع Python `3.11` كبيئة تشغيل افتراضية.

```shell
git clone https://github.com/harry0703/MoneyPrinterTurbo.git
cd MoneyPrinterTurbo
uv python install 3.11
uv sync --frozen
```

إذا كنت لا تستخدم `uv` بعد، فما زال بإمكانك استخدام `venv + pip`.

```shell
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ملاحظات:
- أصبح `pyproject.toml` هو ملف الاعتماديات الأساسي.
- يُثبّت `uv.lock` البيئة المُحدّدة، لذا يُنصح بـ `uv sync --frozen` افتراضياً.
- يُحتفظ بـ `requirements.txt` فقط للتثبيت القديم المعتمد على `pip`.

#### ② تثبيت ImageMagick

###### Windows:

- نزّل من https://imagemagick.org/script/download.php واختر نسخة Windows، وتأكد من اختيار نسخة **المكتبة الساكنة (static library)**، مثل ImageMagick-7.1.1-32-Q16-x64-**static**.exe
- ثبّت ImageMagick الذي نزّلته، **ولا تغيّر مسار التثبيت**
- عدّل ملف الإعدادات `config.toml`، واضبط `imagemagick_path` على مسار التثبيت الفعلي لديك

###### MacOS:

```shell
brew install imagemagick
```

###### Ubuntu

```shell
sudo apt-get install imagemagick
```

###### CentOS

```shell
sudo yum install ImageMagick
```

#### ③ تشغيل واجهة الويب 🌐

لاحظ أنك بحاجة لتنفيذ الأوامر التالية في `المجلد الجذر` لمشروع MoneyPrinterTurbo

###### Windows

```shell
uv run streamlit run ./webui/Main.py --browser.gatherUsageStats=False --server.showEmailPrompt=False
```

إذا كنت قد فعّلت البيئة الافتراضية يدوياً، فما زال بإمكانك تشغيل:

```bat
webui.bat
```

###### MacOS أو Linux

```shell
uv run streamlit run ./webui/Main.py --browser.gatherUsageStats=False --server.showEmailPrompt=False
```

إذا كنت قد فعّلت البيئة الافتراضية يدوياً، فما زال بإمكانك تشغيل:

```shell
sh webui.sh
```

بعد التشغيل، سيُفتح المتصفح تلقائياً

#### ④ تشغيل خدمة الـ API 🚀

```shell
uv run python main.py
```

إذا كنت قد فعّلت البيئة الافتراضية يدوياً، فما زال بإمكانك تشغيل:

```shell
python main.py
```

## شكر خاص 🙏

نظراً لأن **نشر** و**استخدام** هذا المشروع يمثّل عتبةً معينة لبعض المستخدمين المبتدئين، نودّ أن نتقدّم بشكر خاص إلى

**RecCloud (منصة خدمات وسائط متعددة مدعومة بالذكاء الاصطناعي)** لتقديمها خدمة `AI Video Generator` مجانية مبنية على هذا
المشروع. فهي تتيح الاستخدام عبر الإنترنت دون نشر، وهو أمر مريح للغاية.

- النسخة الصينية: https://reccloud.cn
- النسخة الإنجليزية: https://reccloud.com

![](docs/reccloud.com.jpg)

## شكراً للرعاية 🙏

شكراً لـ Picwish https://picwish.com على دعمها ورعايتها لهذا المشروع، مما يتيح التحديث والصيانة المستمرّين.

تركّز Picwish على **مجال معالجة الصور**، وتوفّر مجموعة غنية من **أدوات معالجة الصور** التي تبسّط العمليات المعقّدة إلى حدٍّ بعيد، فتجعل معالجة الصور أسهل حقاً.

![picwish.jpg](docs/picwish.com.jpg)

بعد التشغيل، يمكنك عرض `وثائق الـ API` على http://127.0.0.1:8080/docs واختبار الواجهة مباشرةً عبر الإنترنت
لتجربة سريعة.

## تركيب الصوت 🗣

يمكن عرض قائمة بجميع الأصوات المدعومة هنا: [قائمة الأصوات](./docs/voice-list.txt)

2024-04-16 v1.1.2 أُضيفت 9 أصوات تركيب صوتي جديدة من Azure تتطلب ضبط مفتاح API. هذه الأصوات تبدو أكثر واقعية.

## توليد الترجمة 📜

حالياً، هناك طريقتان لتوليد الترجمة:

- **edge**: سرعة توليد أعلى، وأداء أفضل، ولا متطلبات خاصة لمواصفات الحاسوب، لكن الجودة قد تكون غير مستقرة
- **whisper**: سرعة توليد أبطأ، وأداء أضعف، ومتطلبات خاصة لمواصفات الحاسوب، لكن الجودة أكثر موثوقية

يمكنك التبديل بينهما بتعديل `subtitle_provider` في ملف الإعدادات `config.toml`

يُنصح باستخدام وضع `edge`، والتبديل إلى وضع `whisper` إذا كانت جودة الترجمة المُولّدة غير مُرضية.

> ملاحظة:
>
> 1. في وضع whisper، تحتاج إلى تنزيل ملف نموذج من HuggingFace بحجم نحو 3GB، فتأكد من اتصال إنترنت جيد
> 2. إذا تُرك فارغاً، فهذا يعني أنه لن تُولَّد أي ترجمة.

> بما أن HuggingFace غير متاح في الصين، يمكنك استخدام الطرق التالية لتنزيل ملف نموذج `whisper-large-v3`

روابط التنزيل:

- Baidu Netdisk: https://pan.baidu.com/s/11h3Q6tsDtjQKTjUu3sc5cA?pwd=xjs9
- Quark Netdisk: https://pan.quark.cn/s/3ee3d991d64b

بعد تنزيل النموذج، فُكّ ضغطه وضع المجلد بالكامل في `.\MoneyPrinterTurbo\models`،
وينبغي أن يبدو مسار الملف النهائي هكذا: `.\MoneyPrinterTurbo\models\whisper-large-v3`

```
MoneyPrinterTurbo
  ├─models
  │   └─whisper-large-v3
  │          config.json
  │          model.bin
  │          preprocessor_config.json
  │          tokenizer.json
  │          vocabulary.json
```

## موسيقى الخلفية 🎵

تقع موسيقى خلفية الفيديوهات في مجلد المشروع `resource/songs`.
> يتضمّن المشروع الحالي بعض الموسيقى الافتراضية من فيديوهات YouTube. إن وُجدت مشكلات حقوق نشر، فالرجاء حذفها.

## خطوط الترجمة 🅰

تقع خطوط عرض ترجمة الفيديو في مجلد المشروع `resource/fonts`، ويمكنك أيضاً إضافة خطوطك الخاصة.

## الأسئلة الشائعة 🤔

### ❓RuntimeError: No ffmpeg exe could be found

في الوضع الطبيعي، يُنزَّل ffmpeg ويُكتشَف تلقائياً.
لكن إذا كانت بيئتك تعاني مشكلات تمنع التنزيل التلقائي، فقد تواجه الخطأ التالي:

```
RuntimeError: No ffmpeg exe could be found.
Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.
```

في هذه الحالة، يمكنك تنزيل ffmpeg من https://www.gyan.dev/ffmpeg/builds/ ثم فك ضغطه وضبط `ffmpeg_path` على مسار
التثبيت الفعلي لديك.

```toml
[app]
# الرجاء الضبط بحسب مسارك الفعلي، ولاحظ أن فاصل المسارات في Windows هو \\
ffmpeg_path = "C:\\Users\\harry\\Downloads\\ffmpeg.exe"
```

### ❓ImageMagick is not installed on your computer

[issue 33](https://github.com/harry0703/MoneyPrinterTurbo/issues/33)

1. اتبع `عنوان التنزيل` الموجود في `الإعداد النموذجي`
   لتثبيت https://imagemagick.org/archive/binaries/ImageMagick-7.1.1-30-Q16-x64-static.exe (باستخدام المكتبة الساكنة)
2. لا تثبّت في مسار يحتوي على أحرف صينية لتجنّب مشكلات غير متوقّعة

[issue 54](https://github.com/harry0703/MoneyPrinterTurbo/issues/54#issuecomment-2017842022)

لأنظمة Linux، يمكنك تثبيته يدوياً، راجع https://cn.linux-console.net/?p=16978

شكراً لـ [@wangwenqiao666](https://github.com/wangwenqiao666) على بحثه واستكشافه

### ❓ImageMagick's security policy prevents operations related to temporary file @/tmp/tmpur5hyyto.txt

يمكنك إيجاد هذه السياسات في ملف إعدادات ImageMagick policy.xml.
يقع هذا الملف عادةً في /etc/ImageMagick-`X`/ أو موقع مشابه في مجلد تثبيت ImageMagick.
عدّل المُدخل الذي يحتوي على `pattern="@"`، وغيّر `rights="none"` إلى `rights="read|write"` للسماح بعمليات القراءة والكتابة على الملفات.

### ❓OSError: [Errno 24] Too many open files

تنتج هذه المشكلة عن حدّ النظام لعدد الملفات المفتوحة. يمكنك حلّها بتعديل حدّ فتح الملفات في النظام.

تحقّق من الحدّ الحالي:

```shell
ulimit -n
```

إن كان منخفضاً جداً، يمكنك زيادته، مثلاً:

```shell
ulimit -n 10240
```

### ❓Whisper model download failed, with the following error

LocalEntryNotfoundEror: Cannot find an appropriate cached snapshotfolderfor the specified revision on the local disk and
outgoing trafic has been disabled.
To enablerepo look-ups and downloads online, pass 'local files only=False' as input.

أو

An error occurred while synchronizing the model Systran/faster-whisper-large-v3 from the Hugging Face Hub:
An error happened while trying to locate the files on the Hub and we cannot find the appropriate snapshot folder for the
specified revision on the local disk. Please check your internet connection and try again.
Trying to load the model directly from the local cache, if it exists.

الحل: [اضغط لمعرفة كيفية تنزيل النموذج يدوياً من قرص الشبكة](#توليد-الترجمة-)

## الملاحظات والاقتراحات 📢

- يمكنك إرسال [issue](https://github.com/harry0703/MoneyPrinterTurbo/issues) أو
  [pull request](https://github.com/harry0703/MoneyPrinterTurbo/pulls).

## الرخصة 📝

اضغط لعرض ملف [`LICENSE`](LICENSE)

## تاريخ النجوم (Star History)

[![Star History Chart](https://api.star-history.com/svg?repos=harry0703/MoneyPrinterTurbo&type=Date)](https://star-history.com/#harry0703/MoneyPrinterTurbo&Date)
