<div dir="rtl">

<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# وكيل هيرميس ☤ (Hermes Agent)

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="المستندات"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="الترخيص: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="تطوير Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**وكيل الذكاء الاصطناعي ذاتي التطوير المُبتكر من قِبل [Nous Research](https://nousresearch.com).** إنه الوكيل الوحيد المزوّد بحلقة تعلّم مدمجة — حيث ينشئ المهارات من واقع التجربة، ويُحسّنها أثناء الاستخدام، ويحثّ نفسه على استدامة المعرفة وحفظها، ويبحث في سجل محادثاته السابقة، ويبني نموذجًا متعمقًا يعكس هويتك واحتياجاتك عبر الجلسات المتعددة. يمكنك تشغيله على خادم افتراضي خاص (VPS) بقيمة 5 دولارات، أو على مجموعة معالجات رسومية (GPU cluster)، أو عبر بنية تحتية خالية من الخوادم (serverless) تكاد تنعدم تكلفتها في أوقات الخمول. وهو غير مقيد بحاسوبك المحمول — يمكنك التحدث معه من Telegram بينما يعمل هو على جهاز افتراضي سحابي.

استخدم أي نموذج تريده — [Nous Portal](https://portal.nousresearch.com)، أو [OpenRouter](https://openrouter.ai)، أو OpenAI، أو نقطة النهاية الخاصة بك، والعديد من [المزوّدين الآخرين](https://hermes-agent.nousresearch.com/docs/integrations/providers). بدّل بين النماذج بكل سهولة عبر الأمر `hermes model` — دون أي تعديل في التعليمات البرمجية ودون أي قيود.

<table>
<tr><td><b>واجهة طرفية حقيقية</b></td><td>واجهة TUI متكاملة تدعم التحرير متعدد الأسطر، والإكمال التلقائي لأوامر الشرطة المائلة (slash commands)، وسجل المحادثات، وخاصية المقاطعة وإعادة التوجيه، وبث مخرجات الأدوات لحظيًا.</td></tr>
<tr><td><b>يتواجد أينما تكون</b></td><td>يعمل عبر Telegram و Discord و Slack و WhatsApp و Signal وواجهة الأوامر CLI — كل ذلك من خلال عملية بوابة (gateway) موحدة. ويدعم تفريغ المذكرات الصوتية، واستمرارية المحادثة عبر المنصات المختلفة.</td></tr>
<tr><td><b>حلقة تعلّم متكاملة ومغلقة</b></td><td>ذاكرة يُنظمها الوكيل ذاتيًا مع تنبيهات دورية. إنشاء تلقائي للمهارات بعد تنفيذ المهام المعقدة. وتتحسن المهارات ذاتيًا أثناء الاستخدام. بحث متقدم في الجلسات عبر FTS5 مع تلخيص بنماذج LLM للاسترجاع عبر الجلسات. نمذجة جدلية للمستخدم عبر <a href="https://github.com/plastic-labs/honcho">Honcho</a>. ومتوافق بالكامل مع المعيار المفتوح <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>أتمتة مجدولة</b></td><td>مجدول مهام مدمج (cron scheduler) مع التوصيل لأي منصة. تقارير يومية، ونسخ احتياطية ليلية، وعمليات تدقيق أسبوعية — باللغة الطبيعية بالكامل وتعمل دون الحاجة إلى رقابة.</td></tr>
<tr><td><b>التفويض والتشغيل المتوازي</b></td><td>إطلاق وكلاء فرعيين معزولين لمسارات العمل المتوازية. اكتب نصوص Python برمجية تستدعي الأدوات عبر RPC، لتحويل المهام متعددة الخطوات إلى جولات محادثة دون أي تكلفة على سياق الذاكرة.</td></tr>
<tr><td><b>يعمل في أي مكان، وليس فقط على حاسوبك المحمول</b></td><td>سبع بيئات تشغيل طرفية — محليًا (Local)، و Docker، و SSH، و Singularity، و Modal، و Daytona، وبيئة Vercel Sandbox. توفر كل من Daytona و Modal استمرارية بدون خوادم — حيث يدخل سياق وكيلك في حالة سبات عند الخمول ويستيقظ عند الحاجة، بتكلفة شبه معدومة بين الجلسات. شغّله على VPS بقيمة 5 دولارات أو على عنقود GPU.</td></tr>
<tr><td><b>جاهز للأبحاث والتطوير</b></td><td>توليد مسارات الأداء على دفعات (Batch trajectory generation)، وضغط مسارات الأداء لتدريب الجيل القادم من نماذج استدعاء الأدوات.</td></tr>
</table>

---

## التثبيت السريع

### أنظمة Linux و macOS و WSL2

<div dir="ltr">

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

</div>

### نظام Windows (الأصلي، PowerShell)

> **تنبيه:** يعمل Hermes على نظام Windows الأصلي دون الحاجة إلى WSL — حيث تعمل واجهة CLI والبوابة وواجهة TUI وجميع الأدوات بشكل أصلي. وإذا كنت تفضل استخدام WSL2، فإن الأمر المخصص لـ Linux/macOS أعلاه يعمل هناك أيضًا. هل واجهت مشكلة؟ يُرجى [فتح تذكرة على GitHub](https://github.com/NousResearch/hermes-agent/issues).

شغّل هذا الأمر في PowerShell:

<div dir="ltr">

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

</div>

يُفوّض مثبت المصدر إدارة كل من Python 3.14 و Node.js و npm و ripgrep و FFmpeg واعتماديات Python إلى مدير الحزم PM. وإذا لم يكن Git مثبتًا، فإنه يُنزّل نسخة موثوقة من Git for Windows في متجر أدوات Hermes، دون استبدال نسخة Git الأساسية في نظامك. راجع [طرق التثبيت](https://hermes-agent.nousresearch.com/docs/getting-started/installation) للاطلاع على حزمة MSIX / App Installer المستقلة.

> **Android / Termux:** يتوفر مستودع APT موقّع لأجهزة aarch64، بقناة مستقرة `stable` (إصدارات موثقة) وقناة تجريبية `canary`. تتضمن الحزمة Python و Node.js وواجهة TUI. يُرجى اتباع [دليل Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux) بدلاً من سكريبت التثبيت المكتبي.
>
> **Windows:** نظام Windows الأصلي مدعوم بالكامل — يثبت أمر PowerShell أعلاه كل شيء. وإذا كنت تفضل استخدام WSL2، فإن أمر Linux يعمل هناك أيضًا. يقع مجلد التثبيت الأصلي في Windows تحت `%LOCALAPPDATA%\hermes`؛ بينما يُثبَّت في WSL2 تحت `~/.hermes` كما في Linux.

بعد التثبيت:

<div dir="ltr">

```bash
source ~/.bashrc    # أعد تحميل الطرفية (أو: source ~/.zshrc)
hermes              # ابدأ المحادثة!
```

</div>

### استكشاف الأخطاء وإصلاحها

#### تنبيه Windows Defender أو برامج مكافحة الفيروسات لملف `uv.exe`

إذا وضع برنامج مكافحة الفيروسات (مثل Bitdefender أو Windows Defender وغيرها) ملف `uv.exe` الموجود في مجلد `bin` التابع لـ Hermes (`%LOCALAPPDATA%\hermes\bin\uv.exe`) في الحجر الصحي، فهذا **إنذار كاذب (False Positive)**. الملف هو أداة `uv` المطورة من Astral — وهي أداة إدارة حزم Python المكتوبة بلغة Rust والتي يدمجها Hermes لإدارة بيئة Python الخاصة به. غالبًا ما تضع محركات مكافحة الفيروسات المعتمدة على التعلم الآلي علامة اشتباه على ملفات Rust الثنائية غير الموقعة التي تقوم بتنزيل الحزم وتثبيتها.

**للتحقق من صحة نسختك:**

<div dir="ltr">

```powershell
# ثبّت GitHub CLI إن لزم الأمر
winget install --id GitHub.cli

# سجّل الدخول إلى GitHub
gh auth login

# شغّل أمر التحقق
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

</div>

إذا أظهر التحقق رسالة "Verification succeeded" وطُبعت القيمة `True` في السطر الأخير، فملفك سليم وموثوق تمامًا.

**لإضافة Hermes إلى القائمة الموثوقة (Whitelist):**
- **Windows Defender:** شغّل PowerShell كمسؤول (Run as Admin) ثم نفّذ:
  `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** أضف استثناءً عبر لوحة تحكم Bitdefender (Protection > Antivirus > Settings > Manage Exceptions)
- أضف **المجلد** نفسه إلى القائمة البيضاء وليس بصمة الملف — حيث يُحدّث Hermes أداة `uv` دوريًا وتتغير البصمة مع كل إصدار جديد.

لمزيد من التفاصيل، راجع تقارير مشروع Astral الرسمية: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553)، [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011)، [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## البدء في الاستخدام

<div dir="ltr">

```bash
hermes              # واجهة الأوامر التفاعلية — ابدأ محادثة فورية
hermes model        # اختر مزود نموذج الذكاء الاصطناعي والنموذج
hermes tools        # اضبط الأدوات المفعلة
hermes config set   # عيّن قيم إعدادات محددة
hermes config get   # اعرض قيم إعدادات محددة
hermes gateway      # شغّل بوابة المراسلة (Telegram و Discord وغيرها)
hermes setup        # شغّل معالج الإعداد الكامل (يضبط كل شيء دفعة واحدة)
hermes claw migrate # استورد الإعدادات من OpenClaw (في حال كنت تنتقل منه)
hermes update       # حدّث البرنامج إلى أحدث إصدار
hermes doctor       # افحص النظام وشخّص أي مشكلات
```

</div>

📖 **[المستندات الكاملة →](https://hermes-agent.nousresearch.com/docs/)**

---

## اختصر جمع مفاتيح API — بوابة Nous Portal

يعمل Hermes مع أي مزوّد تفضله دون أي تغيير. ولكن إذا كنت تفضل عدم جمع خمسة مفاتيح API منفصلة للنموذج، والبحث على الويب، وتوليد الصور، وتحويل النص إلى كلام (TTS)، والمتصفح السحابي، فإن **[Nous Portal](https://portal.nousresearch.com)** يغطي كل ذلك باشتراك موحد:

- **أكثر من 300 نموذج** — اختر أيًا منها عبر الأمر `/model <name>`
- **بوابة الأدوات (Tool Gateway)** — البحث على الويب، وتوليد الصور (FAL)، وتحويل النص إلى كلام (OpenAI)، والمتصفح السحابي (Browser Use)، جميعها مفعّلة عبر اشتراكك دون الحاجة لحسابات إضافية.

أمر واحد بعد التثبيت الجديد:

<div dir="ltr">

```bash
hermes setup --portal
```

</div>

يقوم هذا الأمر بتسجيل دخولك عبر OAuth، وتعيين Nous كمزوّد أساسي، وتفعيل بوابة الأدوات. يمكنك التحقق من الأدوات المرتبطة في أي وقت عبر `hermes portal info`. التفاصيل الكاملة في [صفحة توثيق بوابة الأدوات](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

يمكنك دائمًا استخدام مفاتيحك الخاصة لكل أداة متى شئت — فالبوابة تعمل حسب كل خدمة وليست بنظام الكل أو لا شيء.

---

## مرجع سريع: واجهة الأوامر (CLI) مقابل منصات المراسلة

يمتلك Hermes نقطتي دخول: ابدأ واجهة الطرفية باستخدام `hermes`، أو شغّل البوابة وتواصل معه من Telegram أو Discord أو Slack أو WhatsApp أو Signal أو البريد الإلكتروني. بمجرد بدء المحادثة، تتشارك الواجهتان العديد من أوامر الشرطة المائلة (slash commands).

<div dir="ltr">

| الإجراء (Action) | واجهة الأوامر (CLI) | منصات المراسلة (Messaging platforms) |
| :--- | :--- | :--- |
| بدء المحادثة | `hermes` | شغّل `hermes gateway setup` ثم `hermes gateway start`، وأرسل رسالة للبوت |
| بدء محادثة جديدة | `/new` أو `/reset` | `/new` أو `/reset` |
| تغيير النموذج | `/model [provider:model]` | `/model [provider:model]` |
| تعيين شخصية للوكيل | `/personality [name]` | `/personality [name]` |
| إعادة المحاولة أو التراجع | `/retry`, `/undo` | `/retry`, `/undo` |
| ضغط السياق / مراجعة الاستهلاك | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| تصفح المهارات | `/skills` أو `/<skill-name>` | `/<skill-name>` |
| إيقاف العمل الحالي | `Ctrl+C` أو أرسل رسالة جديدة | `/stop` أو أرسل رسالة جديدة |
| حالة المنصة الحالية | `/platforms` | `/status`, `/sethome` |

</div>

للاطلاع على القوائم الكاملة للأوامر، راجع [دليل CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) و [دليل بوابة المراسلة](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## المستندات

تتوفر جميع المستندات والشروحات عبر الرابط **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

<div dir="ltr">

| القسم (Section) | المحتوى (What's Covered) |
| :--- | :--- |
| [البدء السريع (Quickstart)](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | التثبيت ← الإعداد ← أول محادثة في دقيقتين |
| [استخدام CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | الأوامر، اختصارات لوحة المفاتيح، الشخصيات، الجلسات |
| [الإعدادات (Configuration)](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | ملف الإعدادات، المزودون، النماذج، والخيارات الكاملة |
| [بوابة المراسلة (Messaging Gateway)](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram و Discord و Slack و WhatsApp و Signal و Home Assistant |
| [الأمان (Security)](https://hermes-agent.nousresearch.com/docs/user-guide/security) | الموافقة على الأوامر، اقتران الرسائل الخاصة، عزل الحاويات |
| [الأدوات وحزمها (Tools & Toolsets)](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | أكثر من 40 أداة، نظام حزم الأدوات، بيئات الطرفية |
| [نظام المهارات (Skills System)](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | الذاكرة الإجرائية، مركز المهارات، وإنشاء المهارات |
| [الذاكرة (Memory)](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | الذاكرة الدائمة، ملفات تعريف المستخدم، وأفضل الممارسات |
| [تكامل MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | ربط أي خادم MCP لإضافة إمكانيات موسعة |
| [جدولة Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | المهام المجدولة مع التوصيل عبر مختلف المنصات |
| [ملفات السياق (Context Files)](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | سياق المشروع الذي يوجه كل جولة محادثة |
| [الهيكلية البرمجية (Architecture)](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | بنية المشروع، حلقة عمل الوكيل، الفئات الأساسية |
| [المساهمة البرمجية (Contributing)](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | إعداد بيئة التطوير، مسار عمل طلبات الدمج (PR)، أسلوب الكود |
| [مرجع الأوامر (CLI Reference)](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | جميع الأوامر والخيارات بالتفصيل |
| [متغيرات البيئة (Environment Variables)](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | الدليل الكامل لمتغيرات البيئة |

</div>

---

## الانتقال من OpenClaw

إذا كنت قادمًا من OpenClaw، يمكن لـ Hermes استيراد إعداداتك وذكرياتك ومهاراتك ومفاتيح API الخاصة بك تلقائيًا.

**أثناء الإعداد لأول مرة:** يكتشف معالج الإعداد (`hermes setup`) مجلد `~/.openclaw` تلقائيًا ويعرض عليك خيار الترحيل قبل بدء الضبط.

**في أي وقت بعد التثبيت:**

<div dir="ltr">

```bash
hermes claw migrate              # ترحيل تفاعلي (الإعداد المسبق الكامل)
hermes claw migrate --dry-run    # معاينة ما سيتم ترحيله مسبقًا
hermes claw migrate --preset user-data   # ترحيل البيانات دون الأسرار ومفاتيح API
hermes claw migrate --overwrite  # استبدال الملفات الموجودة في حال التعارض
```

</div>

ما يتم استيراده وترحيله:

- **SOUL.md** — ملف شخصية الوكيل
- **الذكريات (Memories)** — مدخلات ملفي MEMORY.md و USER.md
- **المهارات (Skills)** — المهارات المنشأة بواسطة المستخدم ← `~/.hermes/skills/openclaw-imports/`
- **قائمة الأوامر المسموحة (Command allowlist)** — أنماط وأوامر الموافقة
- **إعدادات المراسلة** — إعدادات المنصات، المستخدمون المصرح لهم، مجلد العمل
- **مفاتيح API** — المفاتيح المسموح بها (Telegram و OpenRouter و OpenAI و Anthropic و ElevenLabs)
- **ملفات الصوت (TTS assets)** — ملفات الصوت في مساحة العمل
- **تعليمات مساحة العمل** — ملف AGENTS.md (باستخدام `--workspace-target`)

راجع `hermes claw migrate --help` للاطلاع على جميع الخيارات، أو استخدم مهارة `openclaw-migration` للحصول على عملية ترحيل تفاعلية يقودها الوكيل مع معاينات دقيقة.

---

## المساهمة في المشروع (Contributing)

نرحب بجميع المساهمات! راجع [دليل المساهمة](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) لمعرفة إعداد بيئة التطوير، وأسلوب الكود، ومسار عمل طلبات الدمج (PRs).

راجع [دليل سير عمل مدير الحزم PM](website/docs/reference/package-management.md#developer-workflow) لتفعيل البيئة، واستخدامها اليومي، وإدارة الاعتماديات.
ويغطي [إعداد بيئة التطوير](CONTRIBUTING.md#development-setup) بيئة الاختبار المستقلة وأوامر التحقق.

---

## المجتمع (Community)

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [مركز المهارات (Skills Hub)](https://agentskills.io)
- 🐛 [المشكلات والتذاكر (Issues)](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — خادم MCP للتحكم في سطح مكتب Linux مخصص لـ Hermes ومستضيفي MCP الآخرين، مع أشجار إمكانية الوصول AT-SPI، وإدخال Wayland/X11، والتقاط الشاشة، واستهداف نوافذ مدير العرض.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — جسر WeChat المجتمعي: شغّل Hermes Agent و OpenClaw على نفس حساب WeChat.

---

## الترخيص (License)

ترخيص MIT — راجع ملف [LICENSE](LICENSE).

تم التطوير بواسطة [Nous Research](https://nousresearch.com).

</div>
