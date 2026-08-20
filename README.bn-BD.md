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
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
  <a href="README.bn-BD.md"><img src="https://img.shields.io/badge/Lang-Bangla-blue?style=for-the-badge" alt="বাংলা"></a>
</p>

**[Nous Research](https://nousresearch.com)-এর তৈরি স্ব-উন্নতিশীল (self-improving) AI এজেন্ট।** এটিই একমাত্র এজেন্ট যার নিজস্ব একটি অন্তর্নির্মিত শেখার লুপ রয়েছে— এটি অভিজ্ঞতা থেকে নতুন দক্ষতা তৈরি করে, ব্যবহারের সময় সেগুলো আরও উন্নত করে, জ্ঞান সংরক্ষণের জন্য নিজেকে নিজেই নির্দেশ দেয়, নিজের পুরনো কথোপকথনগুলো খুঁজে বের করে এবং প্রতিটি সেশনের মাধ্যমে আপনার সম্পর্কে আরও গভীর একটি মডেল তৈরি করে।

আপনি এটি মাত্র ৫ ডলারের একটি VPS, একটি GPU ক্লাস্টার, অথবা এমন কোনো সার্ভারলেস ইনফ্রাস্ট্রাকচারে চালাতে পারেন যা নিষ্ক্রিয় (idle) অবস্থায় থাকলে প্রায় কোনো খরচই হয় না। এটি শুধু আপনার ল্যাপটপেই সীমাবদ্ধ নয়— ক্লাউড VM-এ চলাকালীন আপনি Telegram-এর মাধ্যমেও এর সাথে কথা বলতে পারবেন।

আপনার পছন্দমতো যেকোনো মডেল ব্যবহার করুন — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, আপনার নিজস্ব এন্ডপয়েন্ট এবং [আরও অনেকগুলো](https://hermes-agent.nousresearch.com/docs/integrations/providers)। `hermes model` কমান্ড দিয়ে সহজেই পরিবর্তন করুন — কোনো কোড পরিবর্তনের প্রয়োজন নেই, কোনো লক-ইন নেই।

<table>
<tr><td><b>একটি বাস্তব টার্মিনাল ইন্টারফেস</b></td><td>মাল্টিলাইন এডিটিং, স্ল্যাশ-কমান্ড অটো-কমপ্লিট, কথোপকথনের ইতিহাস, ইন্টারাপ্ট-এন্ড-রিডাইরেক্ট এবং স্ট্রিমিং টুল আউটপুটসহ সম্পূর্ণ TUI।</td></tr>
<tr><td><b>আপনার সাথেই থাকে</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, এবং CLI — সবকিছু একটিমাত্র গেটওয়ে প্রসেস থেকে। ভয়েস মেমো ট্রান্সক্রিপশন, ক্রস-প্ল্যাটফর্ম কথোপকথনের ধারাবাহিকতা।</td></tr>
<tr><td><b>একটি ক্লোজড লার্নিং লুপ</b></td><td>নির্দিষ্ট সময় পরপর নজ (nudge) সহ এজেন্ট-পরিচালিত মেমরি। জটিল কাজের পর স্বয়ংক্রিয়ভাবে স্কিল তৈরি। ব্যবহারের সময় স্কিলগুলোর স্ব-উন্নতি। পূর্বের সেশনের কথা মনে করার জন্য LLM সামারাইজেশনসহ FTS5 সেশন সার্চ। <a href="https://github.com/plastic-labs/honcho">Honcho</a> ডায়ালেক্টিক ইউজার মডেলিং। <a href="https://agentskills.io">agentskills.io</a> ওপেন স্ট্যান্ডার্ডের সাথে সামঞ্জস্যপূর্ণ।</td></tr>
<tr><td><b>শিডিউল করা অটোমেশন</b></td><td>যেকোনো প্ল্যাটফর্মে ডেলিভারি সুবিধাসহ বিল্ট-ইন ক্রন (cron) শিডিউলার। ডেইলি রিপোর্ট, নাইটলি ব্যাকআপ, উইকলি অডিট — সবকিছু মানুষের ভাষায়, আপনার উপস্থিতি ছাড়াই চলতে থাকে।</td></tr>
<tr><td><b>ডেলিগেশন এবং প্যারালালাইজেশন</b></td><td>প্যারালাল কাজের জন্য সম্পূর্ণ আলাদা সাব-এজেন্ট তৈরি করতে পারে। RPC-এর মাধ্যমে টুল কল করে এমন Python স্ক্রিপ্ট লিখতে পারে, যা বহুধাপের কাজগুলোকে জিরো-কনটেক্সট-কস্ট টার্নে পরিণত করে।</td></tr>
<tr><td><b>যেকোনো জায়গায় চলে, শুধু আপনার ল্যাপটপে নয়</b></td><td>ছয়টি টার্মিনাল ব্যাকএন্ড — Local, Docker, SSH, Singularity, Modal, এবং Daytona। Daytona এবং Modal সার্ভারলেস পারসিস্টেন্স দেয় — আপনার এজেন্টের এনভায়রনমেন্ট অলস অবস্থায় হাইবারনেট করে এবং প্রয়োজনে জেগে ওঠে, ফলে সেশনগুলোর মাঝে প্রায় কোনো খরচই হয় না। এটি ৫ ডলারের VPS বা GPU ক্লাস্টারেও চালাতে পারেন।</td></tr>
<tr><td><b>রিসার্চ-রেডি</b></td><td>ব্যাচ ট্রাজেক্টরি জেনারেশন, পরবর্তী প্রজন্মের টুল-কলিং মডেলগুলোকে প্রশিক্ষণের জন্য ট্রাজেক্টরি কম্প্রেশন।</td></tr>
</table>

---

## দ্রুত ইনস্টলেশন (Quick Install)

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

```

### Windows (নেটিভ, PowerShell)

> **সতর্কতা:** নেটিভ উইন্ডোজ WSL ছাড়াই Hermes চালায় — CLI, গেটওয়ে, TUI এবং টুলস সবই নেটিভলি কাজ করে। আপনি যদি WSL2 ব্যবহার করতে চান, তবে উপরের Linux/macOS কমান্ডটিও কাজ করবে। কোনো সমস্যা পেলে দয়া করে [ইস্যু সাবমিট করুন](https://github.com/NousResearch/hermes-agent/issues)।

PowerShell-এ এটি রান করুন:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)

```

ইনস্টলার সবকিছু হ্যান্ডেল করে: uv, Python 3.11, Node.js, ripgrep, ffmpeg এবং **একটি পোর্টেবল Git Bash** (MinGit, `%LOCALAPPDATA%\hermes\git`-এ আনপ্যাক করা হয় — কোনো অ্যাডমিন পারমিশন লাগে না, সিস্টেম Git ইনস্টলেশন থেকে সম্পূর্ণ আলাদা)। শেল কমান্ড চালানোর জন্য Hermes এই বান্ডেল করা Git Bash ব্যবহার করে।

আপনার যদি আগে থেকেই Git ইনস্টল করা থাকে, তবে ইনস্টলার সেটি শনাক্ত করে ব্যবহার করবে। অন্যথায় মাত্র ~৪৫MB এর MinGit ডাউনলোডই যথেষ্ট — এটি সিস্টেম Git-এর কোনো পরিবর্তন বা ক্ষতি করবে না।

> **Android / Termux:** পরীক্ষিত ম্যানুয়াল ইনস্টলেশন পদ্ধতি [Termux গাইডে](https://hermes-agent.nousresearch.com/docs/getting-started/termux) দেওয়া আছে। Termux-এ, Hermes একটি নির্দিষ্ট `.[termux]` এক্সট্রা ইনস্টল করে কারণ সম্পূর্ণ `.[all]` এক্সট্রাটি বর্তমানে অ্যান্ড্রয়েডের সাথে বেমানান ভয়েস ডিপেন্ডেন্সি ডাউনলোড করে।
> **Windows:** নেটিভ উইন্ডোজ সম্পূর্ণরূপে সাপোর্টেড — উপরের PowerShell কমান্ডটি সবকিছু ইনস্টল করে। আপনি যদি WSL2 পছন্দ করেন, তবে Linux কমান্ডটিও কাজ করবে। নেটিভ উইন্ডোজের ইনস্টলেশন `%LOCALAPPDATA%\hermes`-এ থাকে; WSL2-এর ক্ষেত্রে এটি লিনাক্সের মতোই `~/.hermes`-এ ইনস্টল হয়।

ইনস্টলেশনের পর:

```bash
source ~/.bashrc    # শেল পুনরায় লোড করুন (অথবা: source ~/.zshrc)
hermes              # কথোপকথন শুরু করুন!

```

### ট্রাবলশুটিং

#### Windows Defender বা অ্যান্টিভাইরাস `uv.exe` কে ম্যালওয়্যার হিসেবে ফ্লাগ করলে

যদি আপনার অ্যান্টিভাইরাস (Bitdefender, Windows Defender ইত্যাদি) Hermes-এর `bin` ফোল্ডার (`%LOCALAPPDATA%\hermes\bin\uv.exe`) থেকে `uv.exe` ফাইলটিকে কোয়ারেন্টাইনে পাঠায়, তবে এটি একটি **ফলস পজিটিভ (false positive)**। ফাইলটি হলো Astral-এর `uv` — এটি একটি Rust-ভিত্তিক Python প্যাকেজ ম্যানেজার যা Hermes তার পাইথন এনভায়রনমেন্ট পরিচালনা করার জন্য ব্যবহার করে। ML-ভিত্তিক অ্যান্টিভাইরাস ইঞ্জিনগুলো প্রায়শই প্যাকেজ ডাউনলোড ও ইনস্টল করা আন-সাইনড (unsigned) Rust বাইনারিগুলোকে ফ্লাগ করে থাকে।

**আপনার কপিটি আসল কি না তা যাচাই করতে:**

```powershell
# প্রয়োজন হলে GitHub CLI ইনস্টল করুন
winget install --id GitHub.cli

# GitHub-এ লগ ইন করুন
gh auth login

# ভেরিফিকেশন রান করুন
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash

```

যদি অ্যাটেস্টেশন বলে "Verification succeeded" এবং শেষ লাইনে `True` প্রিন্ট হয়, তবে সব ঠিক আছে।

**Hermes-কে হোয়াইটলিস্ট করতে:**

* **Windows Defender:** Admin হিসেবে PowerShell রান করুন → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
* **Bitdefender:** Bitdefender কনসোলে একটি এক্সেপশন যোগ করুন (Protection > Antivirus > Settings > Manage Exceptions)
* **ফোল্ডারটি** হোয়াইটলিস্ট করুন, ফাইলের হ্যাশ নয় — Hermes `uv` আপডেট করে এবং প্রতিটি সংস্করণে হ্যাশ পরিবর্তন হয়।

আরও বিস্তারিত জানতে, আপস্ট্রিম Astral রিপোর্টগুলো দেখতে পারেন: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079)।

---

## শুরু করা (Getting Started)

```bash
hermes              # ইন্টারেক্টিভ CLI — কথোপকথন শুরু করুন
hermes model        # আপনার LLM প্রোভাইডার এবং মডেল বেছে নিন
hermes tools        # কোন টুলগুলো এনাবল করা থাকবে তা কনফিগার করুন
hermes config set   # নির্দিষ্ট কনফিগ ভ্যালু সেট করুন
hermes gateway      # মেসেজিং গেটওয়ে চালু করুন (Telegram, Discord ইত্যাদি)
hermes setup        # সম্পূর্ণ সেটআপ উইজার্ড চালান (সবকিছু একসাথে কনফিগার করে)
hermes claw migrate # OpenClaw থেকে মাইগ্রেট করুন (যদি OpenClaw ব্যবহার করে থাকেন)
hermes update       # লেটেস্ট ভার্সনে আপডেট করুন
hermes doctor       # কোনো সমস্যা থাকলে ডায়াগনোজ করুন

```

📖 **[সম্পূর্ণ ডকুমেন্টেশন →](https://hermes-agent.nousresearch.com/docs/)**

---

## API-key সংগ্রহের ঝামেলা এড়ান — Nous Portal

Hermes আপনার পছন্দের যেকোনো প্রোভাইডারের সাথেই কাজ করে — এটা পরিবর্তন হচ্ছে না। তবে আপনি যদি মডেল, ওয়েব সার্চ, ইমেজ জেনারেশন, TTS এবং ক্লাউড ব্রাউজারের জন্য আলাদা পাঁচটি API-key সংগ্রহ করতে না চান, তবে **[Nous Portal](https://portal.nousresearch.com)** একটিমাত্র সাবস্ক্রিপশনের মাধ্যমে এগুলো সব কভার করে:

* **৩০০+ মডেল** — `/model <name>` দিয়ে যেকোনোটি বেছে নিন
* **টুল গেটওয়ে** — ওয়েব সার্চ (Firecrawl), ইমেজ জেনারেশন (FAL), টেক্সট-টু-স্পিচ (OpenAI), ক্লাউড ব্রাউজার (Browser Use), সবকিছু আপনার সাবস্ক্রিপশনের মাধ্যমে রাউট করা হয়। কোনো অতিরিক্ত অ্যাকাউন্টের প্রয়োজন নেই।

নতুন ইনস্টলের পর একটিমাত্র কমান্ড:

```bash
hermes setup --portal

```

এটি আপনাকে OAuth-এর মাধ্যমে লগ ইন করায়, প্রোভাইডার হিসেবে Nous সেট করে এবং টুল গেটওয়ে চালু করে। যেকোনো সময় `hermes portal info` দিয়ে দেখতে পারেন কী কী সংযুক্ত আছে। বিস্তারিত জানতে [টুল গেটওয়ে ডক্স পেজ](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway) দেখুন।

আপনি চাইলে এখনও যেকোনো টুলের জন্য নিজের API-key ব্যবহার করতে পারেন — এই গেটওয়ে ব্যাকএন্ড-ভিত্তিক কাজ করে, সম্পূর্ণ সিস্টেমের ওপর জোর করে চাপিয়ে দেওয়া হয় না।

---

## CLI বনাম মেসেজিং কুইক রেফারেন্স

Hermes-এর দুটি এন্ট্রি পয়েন্ট রয়েছে: `hermes` দিয়ে টার্মিনাল UI চালু করুন, অথবা গেটওয়ে চালু করে Telegram, Discord, Slack, WhatsApp, Signal, বা Email থেকে এর সাথে কথা বলুন। একবার কথোপকথন শুরু হলে, অনেক স্ল্যাশ কমান্ড দুই ইন্টারফেসেই একইভাবে কাজ করে।

| কাজ (Action) | CLI | মেসেজিং প্ল্যাটফর্ম |
| --- | --- | --- |
| কথোপকথন শুরু করুন | `hermes` | `hermes gateway setup` + `hermes gateway start` রান করুন, এরপর বটকে মেসেজ দিন |
| নতুন কথোপকথন শুরু করুন | `/new` বা `/reset` | `/new` বা `/reset` |
| মডেল পরিবর্তন করুন | `/model [provider:model]` | `/model [provider:model]` |
| পার্সোনালিটি সেট করুন | `/personality [name]` | `/personality [name]` |
| শেষ টার্নটি রিট্রাই বা আনডু করুন | `/retry`, `/undo` | `/retry`, `/undo` |
| কনটেক্সট কম্প্রেস করুন / ব্যবহার চেক করুন | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| স্কিল ব্রাউজ করুন | `/skills` বা `/<skill-name>` | `/<skill-name>` |
| চলমান কাজ বন্ধ করুন | `Ctrl+C` অথবা নতুন মেসেজ দিন | `/stop` অথবা নতুন মেসেজ দিন |
| প্ল্যাটফর্ম-ভিত্তিক স্ট্যাটাস | `/platforms` | `/status`, `/sethome` |

সম্পূর্ণ কমান্ড লিস্টের জন্য, [CLI গাইড](https://hermes-agent.nousresearch.com/docs/user-guide/cli) এবং [মেসেজিং গেটওয়ে গাইড](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) দেখুন।

---

## নথিপত্র

সমস্ত নথিপত্র পাওয়া যাবে **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**-এ:

| সেকশন (Section) | বিষয়বস্তু (What's Covered) |
| --- | --- |
| [কুইকস্টার্ট (Quickstart)](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | ইনস্টলেশন → সেটআপ → ২ মিনিটে প্রথম কথোপকথন শুরু করুন |
| [CLI-এর ব্যবহার (CLI Usage)](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | কমান্ড, কীবাইন্ডিং (keybindings), পার্সোনালিটি (personalities), সেশন |
| [কনফিগারেশন (Configuration)](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | কনফিগ ফাইল, প্রোভাইডার, মডেল এবং সমস্ত অপশন |
| [মেসেজিং গেটওয়ে (Messaging Gateway)](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [সিকিউরিটি (Security)](https://hermes-agent.nousresearch.com/docs/user-guide/security) | কমান্ড অ্যাপ্রুভাল, DM পেয়ারিং (pairing), কন্টেইনার আইসোলেশন |
| [টুলস এবং টুলসেট (Tools & Toolsets)](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | ৪০-এর বেশি টুল, টুলসেট সিস্টেম, টার্মিনাল ব্যাকএন্ড |
| [স্কিলস সিস্টেম (Skills System)](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | প্রসিডিউরাল (Procedural) মেমরি, স্কিলস হাব, নতুন স্কিল তৈরি করা |
| [মেমরি (Memory)](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | পারসিস্টেন্ট মেমরি, ইউজার প্রোফাইল, বেস্ট প্র্যাকটিস |
| [MCP ইন্টিগ্রেশন (Integration)](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | বাড়তি ক্ষমতার জন্য যেকোনো MCP সার্ভার কানেক্ট করুন |
| [ক্রন (Cron) শিডিউলিং](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | প্ল্যাটফর্ম ডেলিভারিসহ শিডিউল করা কাজ |
| [কনটেক্সট (Context) ফাইলস](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | প্রজেক্ট কনটেক্সট যা প্রতিটি কথোপকথনকে রূপ দেয় |
| [আর্কিটেকচার (Architecture)](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | প্রজেক্ট স্ট্রাকচার, এজেন্ট লুপ, গুরুত্বপূর্ণ ক্লাসগুলো |
| [কন্ট্রিবিউটিং (Contributing)](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | ডেভেলপমেন্ট সেটআপ, PR প্রসেস, কোডিং স্টাইল |
| [CLI রেফারেন্স (Reference)](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | সমস্ত কমান্ড এবং ফ্ল্যাগ (flags) |
| [এনভায়রনমেন্ট ভেরিয়েবল (Environment Variables)](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | সম্পূর্ণ এনভায়রনমেন্ট ভেরিয়েবল রেফারেন্স |

---

## OpenClaw থেকে মাইগ্রেশন

আপনি যদি OpenClaw থেকে এসে থাকেন, তবে Hermes স্বয়ংক্রিয়ভাবে আপনার সেটিংস, মেমরি, স্কিল এবং API-key গুলো ইমপোর্ট করে নিতে পারে।

**প্রথমবার সেটআপের সময়:** সেটআপ উইজার্ড (`hermes setup`) স্বয়ংক্রিয়ভাবে `~/.openclaw` শনাক্ত করে এবং কনফিগারেশন শুরুর আগেই মাইগ্রেট করার প্রস্তাব দেয়।

**ইনস্টলের পর যেকোনো সময়:**

```bash
hermes claw migrate              # ইন্টারেক্টিভ মাইগ্রেশন (সম্পূর্ণ প্রিসেট)
hermes claw migrate --dry-run    # কী কী মাইগ্রেট হবে তার প্রিভিউ দেখুন
hermes claw migrate --preset user-data   # কোনো সিক্রেট (API-key) ছাড়া মাইগ্রেট করুন
hermes claw migrate --overwrite  # বিদ্যমান কনফ্লিক্টগুলোর ওপর ওভাররাইট করুন

```

**যা যা ইমপোর্ট করা হয়:**

* **SOUL.md** — পারসোনা ফাইল
* **Memories** — MEMORY.md এবং USER.md এন্ট্রিগুলো
* **Skills** — ইউজার-তৈরি স্কিলগুলো → `~/.hermes/skills/openclaw-imports/`
* **Command allowlist** — অ্যাপ্রুভাল প্যাটার্নগুলো
* **Messaging settings** — প্ল্যাটফর্ম কনফিগ, অনুমোদিত ইউজার, ওয়ার্কিং ডিরেক্টরি
* **API keys** — অনুমোদিত সিক্রেটগুলো (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
* **TTS assets** — ওয়ার্কস্পেস অডিও ফাইলগুলো
* **Workspace instructions** — AGENTS.md (`--workspace-target` সহ)

সমস্ত অপশনের জন্য `hermes claw migrate --help` দেখুন, অথবা ড্রাই-রান প্রিভিউসহ ইন্টারেক্টিভ এজেন্ট-পরিচালিত মাইগ্রেশনের জন্য `openclaw-migration` স্কিলটি ব্যবহার করুন।

---

## কন্ট্রিবিউটিং

আমরা কন্ট্রিবিউশন সাদরে গ্রহণ করি! ডেভেলপমেন্ট সেটআপ, কোড স্টাইল এবং PR প্রসেসের জন্য [কন্ট্রিবিউটিং গাইড](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) দেখুন।

কন্ট্রিবিউটারদের জন্য কুইক স্টার্ট — স্ট্যান্ডার্ড ইনস্টলার ব্যবহার করুন, এরপর এটি যে সম্পূর্ণ গিট চেকআউট তৈরি করে `$HERMES_HOME/hermes-agent`-এ (সাধারণত `~/.hermes/hermes-agent`) সেখান থেকে কাজ করুন। এটি `hermes update`, ম্যানেজড venv, লেজি ডিপেন্ডেন্সি, গেটওয়ে এবং ডক্স টুলিংয়ের ব্যবহৃত লেআউটের সাথে সামঞ্জস্যপূর্ণ।

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh

```

ম্যানুয়াল ক্লোন ফলব্যাক (থ্রোঅ্যাওয়ে ক্লোন/CI-এর জন্য যেখানে আপনি ইচ্ছাকৃতভাবে ম্যানেজড ইনস্টল লেআউট চান না):

venv-টি ক্লোন করা সোর্স ট্রি-এর বাইরে তৈরি করুন — এজেন্ট যে ডিরেক্টরি থেকে কাজ করে তার ভেতরে থাকা venv এজেন্টের নিজের চেকআউটের বিরুদ্ধে চালানো কোনো রিলেটিভ-পাথ কমান্ড দিয়ে মুছে যেতে পারে, যা চলমান রানটাইমকে সেশনের মাঝপথেই নষ্ট করে দিতে পারে।

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## কমিউনিটি (Community)

* 💬 [ডিসকর্ড (Discord)](https://discord.gg/NousResearch)
* 📚 [স্কিলস হাব (Skills Hub)](https://agentskills.io)
* 🐛 [ইস্যুস (Issues)](https://github.com/NousResearch/hermes-agent/issues)
* 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Hermes এবং অন্যান্য MCP হোস্টের জন্য লিনাক্স (Linux) ডেস্কটপ-কন্ট্রোল MCP সার্ভার, যাতে AT-SPI অ্যাক্সেসিবিলিটি ট্রি, Wayland/X11 ইনপুট, স্ক্রিনশট এবং কম্পোজিটর উইন্ডো টার্গেটিং রয়েছে।
* 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — কমিউনিটি উইচ্যাট (WeChat) ব্রিজ: একই উইচ্যাট অ্যাকাউন্টে Hermes Agent এবং OpenClaw চালান।

---

## লাইসেন্স (License)

MIT — বিস্তারিত জানতে [LICENSE](LICENSE) দেখুন।

[Nous Research](https://nousresearch.com)-এর তৈরি।