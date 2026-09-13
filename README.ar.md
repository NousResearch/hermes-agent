<div dir="rtl">

<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
  <a href="README.ar.md"><img src="https://img.shields.io/badge/Lang-العربية-teal?style=for-the-badge" alt="العربية"></a>
</p>

**وكيل الذكاء الاصطناعي الذي يطوّر نفسه بنفسه، من بناء [Nous Research](https://nousresearch.com).** إنه الوكيل الوحيد المزوّد بحلقة تعلّم مدمجة (learning loop) — يبتكر مهارات من تجاربه، ويحسّنها أثناء الاستخدام، ويذكّر نفسه بحفظ ما تعلّمه، ويبحث في محادثاته السابقة، ويكوّن عنك، جلسة بعد جلسة، فهمًا أعمق. شغّله على خادم VPS بخمسة دولارات، أو على عنقود GPU‏ (cluster)، أو على بنية بلا خوادم (serverless) لا تكلّف شيئًا يُذكر وهي خاملة. وهو غير مقيّد بجهازك — كلّمه من تيليجرام بينما يعمل على جهاز افتراضي (VM) في السحابة.

استخدم أي نموذج تريده — [Nous Portal](https://portal.nousresearch.com) أو OpenRouter أو OpenAI أو نموذج تستضيفه بنفسك، و[غيرها الكثير](https://hermes-agent.nousresearch.com/docs/integrations/providers). بدّل بينها بأمر <code>hermes model</code> — دون تعديل أي سطر برمجي، ودون ارتباط بمزوّد بعينه.

<table>
<tr><td><b>واجهة طرفية بمعنى الكلمة</b></td><td>واجهة TUI كاملة: تحرير متعدد الأسطر، إكمال تلقائي لأوامر «/»، سجل للمحادثات، قاطعه وغيّر وجهته أثناء العمل، وبث مباشر لمخرجات الأدوات.</td></tr>
<tr><td><b>يعيش في تطبيقاتك اليومية</b></td><td>تيليجرام وديسكورد وسلاك وواتساب وسيجنال وسطر الأوامر (CLI) — كلها من عملية بوابة واحدة. تفريغ نصي للرسائل الصوتية، ومحادثة واحدة متواصلة عبر المنصات.</td></tr>
<tr><td><b>حلقة تعلّم متكاملة (closed loop)</b></td><td>ذاكرة يرعاها الوكيل بنفسه مع تنبيهات دورية. إنشاء تلقائي للمهارات بعد المهام المعقّدة. مهارات تتحسّن ذاتيًا أثناء الاستخدام. بحث في الجلسات عبر FTS5 مع تلخيص بالنماذج اللغوية الكبيرة (LLM) لاستدعاء ما دار في جلسات سابقة. نمذجة حوارية للمستخدم (user modeling) عبر <a href="https://github.com/plastic-labs/honcho">Honcho</a>. متوافق مع معيار <a href="https://agentskills.io">agentskills.io</a> المفتوح.</td></tr>
<tr><td><b>مهام تعمل وحدها في مواعيدها</b></td><td>مجدول مهام (cron) مدمج يوصّل النتائج إلى أي منصة. تقارير يومية، نسخ احتياطي ليلي، تدقيقات أسبوعية — كلها بلغة طبيعية، وتعمل وحدها دون متابعة منك.</td></tr>
<tr><td><b>يوكّل المهام وينجزها بالتوازي</b></td><td>أطلق وكلاء فرعيين معزولين (subagents) لمسارات عمل متوازية. واكتب سكربتات Python تستدعي الأدوات عبر RPC، فتختزل المهام متعددة الخطوات في جولة واحدة بلا أي كلفة على السياق (context).</td></tr>
<tr><td><b>يعمل في أي مكان — لا على جهازك وحده</b></td><td>سبع بيئات تشغيل للطرفية (terminal backends) — محلي وDocker وSSH وSingularity وModal وDaytona وVercel Sandbox. يوفّر Daytona وModal تشغيلًا بلا خوادم (serverless) مع بقاء البيئة — بيئة وكيلك تدخل في سبات عند الخمول وتستيقظ عند الطلب، فلا تكلّف بين الجلسات شيئًا يُذكر. شغّله على خادم VPS بخمسة دولارات أو على عنقود GPU‏ (cluster).</td></tr>
<tr><td><b>جاهز للبحث العلمي</b></td><td>توليد المسارات (trajectories) دفعات، وضغط المسارات لتدريب الجيل القادم من نماذج استدعاء الأدوات.</td></tr>
</table>

---

## التثبيت السريع

### Linux، macOS، WSL2، Termux

<div dir="ltr">

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

</div>

### Windows (تشغيل أصلي عبر PowerShell)

> **تنبيه:** يعمل Hermes على Windows الأصلي دون WSL — سطر الأوامر (CLI) والبوابة وواجهة TUI والأدوات كلها تعمل أصلًا. وإن كنت تفضّل WSL2، فأمر لينكس/ماك أعلاه يعمل هناك أيضًا. وجدت خللًا؟ [افتح بلاغًا (issue)](https://github.com/NousResearch/hermes-agent/issues).

شغّل هذا الأمر في PowerShell:

<div dir="ltr">

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

</div>

يتكفّل المثبّت بكل شيء: uv وPython 3.11 وNode.js وripgrep وffmpeg، **ونسخة محمولة من Git Bash** ‏(MinGit، تُفكّ في <code>%LOCALAPPDATA%\hermes\git</code> — لا تحتاج صلاحيات مسؤول، ومعزولة تمامًا عن أي Git مثبّت على النظام). يستخدم Hermes نسخة Git Bash المرفقة هذه لتشغيل أوامر الصدفة (shell).

إذا كان Git مثبّتًا لديك أصلًا، فسيكتشفه المثبّت ويستخدمه بدلًا منها. وإلا فكل ما تحتاجه تنزيل MinGit بحجم ~45MB — ولن يمسّ أي Git على نظامك أو يتعارض معه.

> **أندرويد / Termux:** الطريقة اليدوية المجرّبة موثّقة في [دليل Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux). على Termux يثبّت Hermes حزمة <code>.[termux]</code> منتقاة، لأن حزمة <code>.[all]</code> الكاملة تسحب حاليًا اعتماديات صوتية غير متوافقة مع أندرويد.
>
> **Windows:** الدعم الأصلي لـ Windows كامل — أمر PowerShell أعلاه يثبّت كل شيء. وإن كنت تفضّل WSL2، فأمر لينكس يعمل هناك. التثبيت الأصلي على Windows يكون في <code>%LOCALAPPDATA%\hermes</code>؛ أما على WSL2 ففي <code>~/.hermes</code> كما في لينكس.

بعد التثبيت:

<div dir="ltr">

```bash
source ~/.bashrc    # أعد تحميل الصدفة (أو: source ~/.zshrc)
hermes              # ابدأ الحديث!
```

</div>

### حل المشكلات

#### يصنّف Windows Defender أو مضاد الفيروسات ملف <code>uv.exe</code> برمجيةً خبيثة

إذا حجر مضاد الفيروسات لديك (Bitdefender أو Windows Defender أو غيرهما) ملف <code>uv.exe</code> من مجلد <code>bin</code> الخاص بـ Hermes‏ (<code>%LOCALAPPDATA%\hermes\bin\uv.exe</code>)، فهذا **إنذار كاذب (false positive)**. الملف هو أداة <code>uv</code> من Astral — مدير حزم Python المكتوب بلغة Rust والذي يرفقه Hermes لإدارة بيئة Python الخاصة به. محركات مضادات الفيروسات المعتمدة على تعلّم الآلة تصنّف عادةً ثنائيات Rust غير الموقّعة التي تنزّل الحزم وتثبّتها.

**للتحقق من أصالة نسختك:**

<div dir="ltr">

```powershell
# ثبّت GitHub CLI إن لم يكن لديك
winget install --id GitHub.cli

# سجّل الدخول إلى GitHub
gh auth login

# شغّل التحقق
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

إذا أظهر التحقق "Verification succeeded" وطبع السطر الأخير <code>True</code>، فنسختك سليمة.

**لإضافة Hermes إلى الاستثناءات:**

- **Windows Defender:** شغّل PowerShell بصلاحيات المسؤول ← <code>Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"</code>
- **Bitdefender:** أضف استثناء من لوحة Bitdefender‏ (Protection > Antivirus > Settings > Manage Exceptions)
- أضف **المجلد** إلى الاستثناءات، لا تجزئة (hash) الملف — يحدّث Hermes أداة uv فتتغيّر التجزئة مع كل إصدار

لمزيد من السياق، راجع البلاغات في مستودع Astral: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553)، [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011)، [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## ابدأ من هنا

<div dir="ltr">

```bash
hermes              # سطر أوامر تفاعلي — ابدأ محادثة
hermes model        # اختر مزوّد النماذج والنموذج
hermes tools        # حدد الأدوات المفعّلة
hermes config set   # اضبط قيم الإعدادات فرادى
hermes config get   # اعرض قيم الإعدادات فرادى
hermes gateway      # شغّل بوابة المراسلة (تيليجرام وديسكورد وغيرهما)
hermes setup        # شغّل معالج الإعداد الكامل (يضبط كل شيء دفعة واحدة)
hermes claw migrate # انتقل من OpenClaw (إن كنت قادمًا منه)
hermes update       # حدّث إلى أحدث إصدار
hermes doctor       # شخّص أي مشكلة
```

</div>

📖 **[الوثائق الكاملة ←](https://hermes-agent.nousresearch.com/docs/)**

---

## وفّر على نفسك جمع مفاتيح API — Nous Portal

يعمل Hermes مع أي مزوّد تختاره — وهذا لن يتغيّر. لكن إن كنت لا تريد جمع خمسة مفاتيح API منفصلة للنموذج والبحث في الويب وتوليد الصور وتحويل النص إلى كلام (TTS) والمتصفح السحابي، فإن **[Nous Portal](https://portal.nousresearch.com)** يغطيها كلها ضمن اشتراك واحد:

- **أكثر من 300 نموذج** — اختر أيًا منها بالأمر <code>/model &lt;name&gt;</code>
- **بوابة الأدوات (Tool Gateway)** — بحث في الويب (Firecrawl)، وتوليد الصور (FAL)، وتحويل النص إلى كلام (OpenAI)، ومتصفح سحابي (Browser Use)، كلها عبر اشتراكك. دون أي حسابات إضافية.

أمر واحد يكفي بعد تثبيت جديد:

<div dir="ltr">

```bash
hermes setup --portal
```

</div>

يسجّل هذا دخولك عبر OAuth، ويجعل Nous مزوّدك، ويفعّل بوابة الأدوات. تحقق مما هو مربوط في أي وقت بالأمر <code>hermes portal info</code>. التفاصيل الكاملة في [صفحة وثائق Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

ويمكنك دائمًا استخدام مفاتيحك الخاصة لأي أداة — فالبوابة تعمل لكل خدمة على حدة، لا بمنطق الكل أو لا شيء.

---

## مرجع سريع: سطر الأوامر (CLI) مقابل المراسلة

لـ Hermes مدخلان: شغّل واجهة الطرفية بالأمر <code>hermes</code>، أو شغّل البوابة وكلّمه من تيليجرام أو ديسكورد أو سلاك أو واتساب أو سيجنال أو البريد الإلكتروني. وبمجرد دخولك في محادثة، كثير من أوامر «/» مشتركة بين الواجهتين.

<div dir="ltr">

| الإجراء | CLI | منصات المراسلة |
| --- | --- | --- |
| ابدأ الحديث | `hermes` | شغّل `hermes gateway setup` ثم `hermes gateway start`، ثم أرسل رسالة للبوت |
| ابدأ محادثة جديدة | `/new` أو `/reset` | `/new` أو `/reset` |
| غيّر النموذج | `/model [provider:model]` | `/model [provider:model]` |
| عيّن شخصية (personality) | `/personality [name]` | `/personality [name]` |
| أعد المحاولة أو تراجع عن آخر جولة | `/retry`، `/undo` | `/retry`، `/undo` |
| اضغط السياق / تحقق من الاستهلاك | `/compress`، `/usage`، `/insights [--days N]` | `/compress`، `/usage`، `/insights [days]` |
| تصفّح المهارات | `/skills` أو `/<skill-name>` | `/<skill-name>` |
| قاطع العمل الجاري | اضغط `Ctrl+C` أو أرسل رسالة جديدة | `/stop` أو أرسل رسالة جديدة |
| حالة كل منصة | `/platforms` | `/status`، `/sethome` |

</div>

للاطلاع على قوائم الأوامر الكاملة، راجع [دليل CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) و[دليل بوابة المراسلة](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## الوثائق

كل الوثائق على **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

<div dir="ltr">

| القسم | ما يغطيه |
| --- | --- |
| [البدء السريع](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | التثبيت ← الإعداد ← أول محادثة في دقيقتين |
| [استخدام CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | الأوامر، اختصارات المفاتيح، الشخصيات، الجلسات |
| [الإعدادات](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | ملف الإعدادات، المزوّدون، النماذج، كل الخيارات |
| [بوابة المراسلة](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | تيليجرام، ديسكورد، سلاك، واتساب، سيجنال، Home Assistant |
| [الأمان](https://hermes-agent.nousresearch.com/docs/user-guide/security) | الموافقة على الأوامر، اقتران الرسائل الخاصة (DM pairing)، عزل الحاويات |
| [الأدوات وأطقمها](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | أكثر من 40 أداة، نظام أطقم الأدوات، بيئات تشغيل الطرفية |
| [نظام المهارات](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | الذاكرة الإجرائية، Skills Hub، إنشاء المهارات |
| [الذاكرة](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | ذاكرة دائمة، ملفات المستخدمين، أفضل الممارسات |
| [تكامل MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | اربط أي خادم MCP لتوسيع القدرات |
| [جدولة cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | مهام مجدولة تُسلَّم نتائجها إلى المنصات |
| [ملفات السياق](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | سياق المشروع الذي يشكّل كل محادثة |
| [البنية المعمارية](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | هيكل المشروع، حلقة الوكيل، الأصناف (classes) الرئيسية |
| [المساهمة](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | إعداد بيئة التطوير، آلية الـ PR، أسلوب الكود |
| [مرجع CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | كل الأوامر والرايات (flags) |
| [متغيرات البيئة](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | مرجع كامل لمتغيرات البيئة |

</div>

---

## الانتقال من OpenClaw

إذا كنت قادمًا من OpenClaw، يستطيع Hermes استيراد إعداداتك وعناصر ذاكرتك ومهاراتك ومفاتيح API تلقائيًا.

**أثناء الإعداد الأول:** يكتشف معالج الإعداد (<code>hermes setup</code>) مجلد <code>~/.openclaw</code> تلقائيًا ويعرض عليك الانتقال قبل بدء الضبط.

**وفي أي وقت بعد التثبيت:**

<div dir="ltr">

```bash
hermes claw migrate              # انتقال تفاعلي (الحزمة الكاملة)
hermes claw migrate --dry-run    # عاين ما الذي سيُنقل
hermes claw migrate --preset user-data   # انتقل دون الأسرار (secrets)
hermes claw migrate --overwrite  # اكتب فوق الملفات المتعارضة
```

</div>

ما الذي يُستورد:

- **SOUL.md** — ملف الشخصية (persona)
- **الذاكرة (Memories)** — مدخلات MEMORY.md وUSER.md
- **المهارات (Skills)** — المهارات التي أنشأها المستخدم ← <code>~/.hermes/skills/openclaw-imports/</code>
- **قائمة الأوامر المسموحة (allowlist)** — أنماط الموافقة
- **إعدادات المراسلة** — إعدادات المنصات، المستخدمون المسموح لهم، مجلد العمل
- **مفاتيح API** — الأسرار المسموح بها (تيليجرام، OpenRouter، OpenAI، Anthropic، ElevenLabs)
- **أصول TTS** — الملفات الصوتية في مساحة العمل
- **تعليمات مساحة العمل** — AGENTS.md (مع <code>--workspace-target</code>)

راجع <code>hermes claw migrate --help</code> لكل الخيارات، أو استخدم مهارة <code>openclaw-migration</code> لانتقال تفاعلي يقوده الوكيل مع معاينات تجريبية (dry-run).

---

## المساهمة

نرحّب بمساهماتك! راجع [دليل المساهمة](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) لإعداد بيئة التطوير وأسلوب الكود وآلية الـ PR.

بداية سريعة للمساهمين — استخدم المثبّت القياسي، ثم اعمل من نسخة git الكاملة التي ينشئها في <code>$HERMES_HOME/hermes-agent</code> (عادةً <code>~/.hermes/hermes-agent</code>). هذا يطابق التخطيط الذي يستخدمه <code>hermes update</code> وبيئة venv المُدارة والاعتماديات الكسولة (lazy) والبوابة وأدوات الوثائق.

<div dir="ltr">

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

</div>

بديل الاستنساخ اليدوي (لنسخ مؤقتة أو بيئات CI لا تريد فيها عمدًا تخطيط التثبيت المُدار):

أنشئ بيئة venv خارج شجرة المصدر المستنسخة — فبيئة venv داخل المجلد الذي يعمل الوكيل منه قد تُمحى بأمر مسار نسبي يشغّله الوكيل على نسخته، فيدمّر بيئة التشغيل وهي تعمل في منتصف الجلسة.

<div dir="ltr">

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

</div>

---

## المجتمع

- 💬 [ديسكورد (Discord)](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [البلاغات (Issues)](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — خادم MCP للتحكم بسطح مكتب لينكس، لـ Hermes وغيره من مضيفي MCP، مع أشجار الوصولية AT-SPI، وإدخال Wayland/X11، ولقطات الشاشة، واستهداف النوافذ عبر المركّب (compositor).
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — جسر WeChat من المجتمع: شغّل Hermes Agent وOpenClaw على حساب WeChat واحد.

---

## الرخصة

MIT — راجع [LICENSE](LICENSE).

من بناء [Nous Research](https://nousresearch.com).

</div>
