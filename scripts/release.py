#!/usr/bin/env python3
"""Hermes Agent Release Script

Generates changelogs and creates GitHub releases with CalVer tags.

Usage:
    # Preview changelog (dry run)
    python scripts/release.py

    # Preview with semver bump
    python scripts/release.py --bump minor

    # Create the release
    python scripts/release.py --bump minor --publish

    # First release (no previous tag)
    python scripts/release.py --bump minor --publish --first-release

    # Override CalVer date (e.g. for a belated release)
    python scripts/release.py --bump minor --publish --date 2026.3.15
"""

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "hermes_cli" / "__init__.py"
PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"

# ──────────────────────────────────────────────────────────────────────
# Git email → GitHub username mapping
# ──────────────────────────────────────────────────────────────────────

# Auto-extracted from noreply emails + manual overrides
AUTHOR_MAP = {
    # teknium (multiple emails)
    "teknium1@gmail.com": "teknium1",
    "0x.badfriend@gmail.com": "discodirector",
    "altriatree@gmail.com": "TruaShamu",
    "m@mobrienv.dev": "mikeyobrien",
    "qiyin.zuo@pcitc.com": "qiyin-code",
    "oleksii.lisikh@gmail.com": "olisikh",
    "leone.parise@gmail.com": "leoneparise",
    "buraysandro9@gmail.com": "ygd58",
    "teknium@nousresearch.com": "teknium1",
    "piyushvp1@gmail.com": "thelumiereguy",
    "harish.kukreja@gmail.com": "counterposition",
    "cleo@edaphic.xyz": "curiouscleo",
    "hirokazu.ogawa@kwansei.ac.jp": "hrkzogw",
    "datapod.k@gmail.com": "dandacompany",
    "127238744+teknium1@users.noreply.github.com": "teknium1",
    "128259593+Gutslabs@users.noreply.github.com": "Gutslabs",
    "50326054+nocturnum91@users.noreply.github.com": "nocturnum91",
    "223003280+Abd0r@users.noreply.github.com": "Abd0r",
    "ra2157218@gmail.com": "Abd0r",
    "abdielv@proton.me": "AJV20",
    "mason@growagainorchids.com": "masonjames",
    "ytchen0719@gmail.com": "liquidchen",
    "am@studio1.tailb672fe.ts.net": "subtract0",
    "axmaiqiu@gmail.com": "qWaitCrypto",
    "maksesipov@gmail.com": "Qwinty",
    "denisamania@gmail.com": "CalmProton",
    "308068+mbac@users.noreply.github.com": "mbac",
    "wesleysimplicio@live.com": "wesleysimplicio",
    "matthew.dean.cater@gmail.com": "SiliconID",
    "xieniu@proton.me": "xieNniu",
    "rw8143a@american.edu": "wali-reheman",
    "egitimviscara@gmail.com": "uzunkuyruk",
    "zhekinmaksim@gmail.com": "Zhekinmaksim",
    "obafemiferanmi1999@gmail.com": "KvnGz",
    "159539633+MottledShadow@users.noreply.github.com": "MottledShadow",
    "aludwin+gh@gmail.com": "adamludwin",
    "ngusev@astralinux.ru": "NikolayGusev-astra",
    "liuguangyong201@hellobike.com": "liuguangyong93",
    "2093036+exiao@users.noreply.github.com": "exiao",
    "20nik.nosov21@gmail.com": "nik1t7n",
    "thunderggnn@gmail.com": "ggnnggez",
    "haozhe4547@gmail.com": "ehz0ah",
    "kevyan1998@gmail.com": "kyan12",
    "rylen.anil@gmail.com": "rylena",
    "godnanijatin@gmail.com": "jatingodnani",
    "252811164+adybag14-cyber@users.noreply.github.com": "adybag14-cyber",
    "14046872+tmimmanuel@users.noreply.github.com": "tmimmanuel",
    "112875006+donramon77@users.noreply.github.com": "donramon77",
    "657290301@qq.com": "IMHaoyan",
    "revar@users.noreply.github.com": "revaraver",
    "dengtaoyuan@dengtaoyuandeMac-mini.local": "dengtaoyuan450-a11y",
    "ysfalweshcan@gmail.com": "Junass1",
    "bartokmagic@proton.me": "Bartok9",
    "androidhtml@yandex.com": "hllqkb",
    "25840394+Bongulielmi@users.noreply.github.com": "Bongulielmi",
    "jonathan.troyer@overmatch.com": "JTroyerOvermatch",
    "harryykyle1@gmail.com": "hharry11",
    "wysie@users.noreply.github.com": "wysie",
    "jkausel@gmail.com": "jkausel-ai",
    "e.silacandmr@gmail.com": "Es1la",
    "51599529+stephen0110@users.noreply.github.com": "stephen0110",
    "265632032+sonic-netizen@users.noreply.github.com": "sonic-netizen",
    "82531659+mwnickerson@users.noreply.github.com": "mwnickerson",
    "sandrohub013@gmail.com": "SandroHub013",
    "maciekczech@users.noreply.github.com": "maciekczech",
    "154585401+LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "zjtan1@gmail.com": "zeejaytan",
    "asslaenn5@gmail.com": "Aslaaen",
    "trae.anderson17@icloud.com": "Tkander1715",
    "beardthelion@users.noreply.github.com": "beardthelion",
    "tangyuanjc@JCdeAIfenshendeMac-mini.local": "tangyuanjc",
    "leon@agentlinker.ai": "agentlinker",
    "santoshhumagain1887@gmail.com": "npmisantosh",
    "novax635@gmail.com": "novax635",
    "krionex1@gmail.com": "Krionex",
    "rxdxxxx@users.noreply.github.com": "rxdxxxx",
    "ma.haohao2@xydigit.com": "MaHaoHao-ch",
    "29756950+revaraver@users.noreply.github.com": "revaraver",
    "nexus@eptic.me": "TheEpTic",
    "74554762+wmagev@users.noreply.github.com": "wmagev",
    "ashermorse@icloud.com": "ashermorse",
    "happy5318@users.noreply.github.com": "happy5318",
    "anatoliygranichenko@gmail.com": "wabrent",
    "cash.williams@acquia.com": "CashWilliams",
    "chengoak@users.noreply.github.com": "chengoak",
    "mrhanoi@outlook.com": "qxxaa",
    "guillaume.meyer@outlook.com": "guillaumemeyer",
    "emelyanenko.kirill@gmail.com": "EmelyanenkoK",
    "lazycat.manatee@gmail.com": "manateelazycat",
    "bzarnitz13@gmail.com": "Beandon13",
    "tony@tonysimons.dev": "asimons81",
    "jetha@google.com": "jethac",
    "jani@0xhoneyjar.xyz": "deep-name",
    "xiangyong@zspace.cn": "CES4751",
    "harish.kukreja@gmail.com": "counterposition",
    "35294173+Fearvox@users.noreply.github.com": "Fearvox",
    "hypnus.yuan@gmail.com": "Hypnus-Yuan",
    "15558128926@qq.com": "xsfX20",
    "binhnt.ht.92@gmail.com": "binhnt92",
    "johnny@Jons-MBA-M4.local": "acesjohnny",
    "1581133593@qq.com": "liu-collab",
    "haidaoe@proton.me": "haidao1919",
    "50561768+zhanggttry@users.noreply.github.com": "zhanggttry",
    "formulahendry@gmail.com": "formulahendry",
    "93757150+bogerman1@users.noreply.github.com": "bogerman1",
    "132852777+rob-maron@users.noreply.github.com": "rob-maron",
    # Matrix parity salvage batch (April 2026)
    "sr@samirusani": "samrusani",
    "angelclaw@AngelMacBook.local": "angel12",
    "charles@cryptoassetrecovery.com": "charles-brooks",
    # DeepSeek v4 + Kimi thinking-mode reasoning_content salvage (April 2026)
    "luwinyang@deepseek.com": "lsdsjy",
    "season.saw@gmail.com": "season179",
    "heathley@Heathley-MacBook-Air.local": "heathley",
    "maliyldzhn@gmail.com": "heathley",
    "vlad19@gmail.com": "dandaka",
    "adamrummer@gmail.com": "cyclingwithelephants",
    # Temporary tool-progress cleanup salvage (May 2026)
    "Mrcharlesiv@gmail.com": "mrcharlesiv",
    "nbot@liizfq.top": "liizfq",
    "274096618+hermes-agent-dhabibi@users.noreply.github.com": "dhabibi",
    "dejie.guo@gmail.com": "JayGwod",
    "133716830+0xKingBack@users.noreply.github.com": "0xKingBack",
    "daixin1204@gmail.com": "SimbaKingjoe",
    "maxence@groine.fr": "MaxyMoos",
    "61830395+leprincep35700@users.noreply.github.com": "leprincep35700",
    # OpenViking viking_read salvage (April 2026)
    "hitesh@gmail.com": "htsh",
    "pty819@outlook.com": "pty819",
    "pty819@users.noreply.github.com": "pty819",
    "517024110@qq.com": "chennest",
    # Curator fixes (Apr 30 2026)
    "yuxiangl490@gmail.com": "y0shua1ee",
    "manmit0x@gmail.com": "0xDevNinja",
    "stevekelly622@gmail.com": "steezkelly",
    "momowind@gmail.com": "momowind",
    "clockwork-codex@users.noreply.github.com": "misery-hl",
    "207811921+misery-hl@users.noreply.github.com": "misery-hl",
    "20nik.nosov21@gmail.com": "nik1t7n",
    "90299797+nik1t7n@users.noreply.github.com": "nik1t7n",
    "suncokret@protonmail.com": "suncokret12",
    "mio.imoto.ai@gmail.com": "mioimotoai-lgtm",
    "aamirjawaid@microsoft.com": "heyitsaamir",
    "johnnncenaaa77@gmail.com": "johnncenae",
    "thomasjhon6666@gmail.com": "ThomassJonax",
    "focusflow.app.help@gmail.com": "yes999zc",
    "rob@atlas.lan": "rmoen",
    # Slack ephemeral slash-ack salvage (May 2026)
    "probepark@users.noreply.github.com": "probepark",
    # Slack batch salvage (May 2026)
    "280484231+prive-fe-bot@users.noreply.github.com": "priveperfumes",
    "amr@ghanem.sa": "amroessam",
    "paperlantern.agent@gmail.com": "Hinotoi-agent",
    "valda@underscore.jp": "valda",
    "162235745+0z1-ghb@users.noreply.github.com": "0z1-ghb",
    "yes999zc@163.com": "yes999zc",
    "343873859@qq.com": "DrStrangerUJN",
    "252818347@qq.com": "hejuntt1014",
    "uzmpsk.dilekakbas@gmail.com": "dlkakbs",
    "beliefanx@gmail.com": "BeliefanX",
    "changchun989@proton.me": "changchun989",
    "jefferson@heimdallstrategy.com": "Mind-Dragon",
    "44753291+Nanako0129@users.noreply.github.com": "Nanako0129",
    "steve.westerhouse@origami-analytics.com": "westers",
    "yeyitech@users.noreply.github.com": "yeyitech",
    "260878550+beenherebefore@users.noreply.github.com": "beenherebefore",
    "79389617+txbxxx@users.noreply.github.com": "txbxxx",
    "liuhao03@bilibili.com": "liuhao1024",
    "130918800+devorun@users.noreply.github.com": "devorun",
    "surat.s@itm.kmutnb.ac.th": "beesrsj2500",
    "beesr@bee.localdomain": "beesrsj2500",
    "mind-dragon@nous.research": "Mind-Dragon",
    "juntingpublic@gmail.com": "JustinUssuri",
    "mtf201013@gmail.com": "ma-pony",
    "sonoyuncudmr@gmail.com": "Sonoyunchu",
    "43525405+yatesjalex@users.noreply.github.com": "yatesjalex",
    "maks.mir@yahoo.com": "say8hi",
    "27719690+Mirac1eSky@users.noreply.github.com": "Mirac1eSky",
    "web3blind@users.noreply.github.com": "web3blind",
    "julia@alexland.us": "alexg0bot",
    "christian@scheid.tech": "scheidti",
    # Moonshot schema anyOf+enum salvage (May 2026)
    "git@local.invalid": "hendrixfreire",
    "1060770+benjaminsehl@users.noreply.github.com": "benjaminsehl",
    "nerijusn76@gmail.com": "Nerijusas",
    # Compaction salvage batch (May 2026)
    "MacroAnarchy@users.noreply.github.com": "MacroAnarchy",
    "itonov@proton.me": "Ito-69",
    "glesstech@gmail.com": "georgeglessner",
    "maxim.smetanin@gmail.com": "maxims-oss",
    "nazirulhafiy@gmail.com": "nazirulhafiy",
    "CREWorx@users.noreply.github.com": "BadTechBandit",
    "yoimexex@gmail.com": "Yoimex",
    "6548898+romanornr@users.noreply.github.com": "romanornr",
    "foxion37@gmail.com": "foxion37",
    "bloodcarter@gmail.com": "bloodcarter",
    "scott@scotttrinh.com": "scotttrinh",
    "quocanh261997@gmail.com": "quocanh261997",
    # contributors (from noreply pattern)
    "david.vv@icloud.com": "davidvv",
    "wangqiang@wangqiangdeMac-mini.local": "xiaoqiang243",
    "snreynolds2506@gmail.com": "snreynolds",
    "35742124+0xbyt4@users.noreply.github.com": "0xbyt4",
    "71184274+MassiveMassimo@users.noreply.github.com": "MassiveMassimo",
    "massivemassimo@users.noreply.github.com": "MassiveMassimo",
    "82637225+kshitijk4poor@users.noreply.github.com": "kshitijk4poor",
    "keifergu@tencent.com": "keifergu",
    "kshitijk4poor@users.noreply.github.com": "kshitijk4poor",
    "SHL0MS@users.noreply.github.com": "SHL0MS",
    "abner.the.foreman@agentmail.to": "Abnertheforeman",
    "adam.manning@pro-serveinc.com": "amanning3390",
    "thomasgeorgevii09@gmail.com": "tochukwuada",
    "sb@wmc.sh": "zicochaos",
    "harryykyle1@gmail.com": "hharry11",
    "kshitijk4poor@gmail.com": "kshitijk4poor",
    "1294707+Tosko4@users.noreply.github.com": "Tosko4",
    "keira.voss94@gmail.com": "keiravoss94",
    "16443023+stablegenius49@users.noreply.github.com": "stablegenius49",
    "fqsy1416@gmail.com": "EKKOLearnAI",
    "octo-patch@github.com": "octo-patch",
    "math0r-be@github.com": "math0r-be",
    "simbamax99@gmail.com": "simbam99",
    "iris@growthpillars.co": "irispillars",
    "185121704+stablegenius49@users.noreply.github.com": "stablegenius49",
    "101283333+batuhankocyigit@users.noreply.github.com": "batuhankocyigit",
    "255305877+ismell0992-afk@users.noreply.github.com": "ismell0992-afk",
    "cyprian@ironin.pl": "iRonin",
    "valdi.jorge@gmail.com": "jvcl",
    "q19dcp@gmail.com": "aj-nt",
    "ebukau84@gmail.com": "UgwujaGeorge",
    "francip@gmail.com": "francip",
    "omni@comelse.com": "omnissiah-comelse",
    "oussama.redcode@gmail.com": "mavrickdeveloper",
    "126368201+vilkasdev@users.noreply.github.com": "vilkasdev",
    "137614867+cutepawss@users.noreply.github.com": "cutepawss",
    "96793918+memosr@users.noreply.github.com": "memosr",
    "mehmet.sr35@gmail.com": "memosr",
    "milkoor@users.noreply.github.com": "milkoor",
    "xuerui911@gmail.com": "Fatty911",
    "131039422+SHL0MS@users.noreply.github.com": "SHL0MS",
    "77628552+raulvidis@users.noreply.github.com": "raulvidis",
    "145567217+Aum08Desai@users.noreply.github.com": "Aum08Desai",
    "256820943+kshitij-eliza@users.noreply.github.com": "kshitij-eliza",
    "jiechengwu@pony.ai": "Jason2031",
    "44278268+shitcoinsherpa@users.noreply.github.com": "shitcoinsherpa",
    "104278804+Sertug17@users.noreply.github.com": "Sertug17",
    "112503481+caentzminger@users.noreply.github.com": "caentzminger",
    "258577966+voidborne-d@users.noreply.github.com": "voidborne-d",
    "3820588+ddupont808@users.noreply.github.com": "ddupont808",
    "liusway405@gmail.com": "voidborne-d",
    "xydarcher@uestc.edu.cn": "Readon",
    "sir_even@icloud.com": "sirEven",
    "36056348+sirEven@users.noreply.github.com": "sirEven",
    "70424851+insecurejezza@users.noreply.github.com": "insecurejezza",
    "jezzahehn@gmail.com": "JezzaHehn",
    "barnacleboy.jezzahehn@agentmail.to": "JezzaHehn",
    "254021826+dodo-reach@users.noreply.github.com": "dodo-reach",
    "259807879+Bartok9@users.noreply.github.com": "Bartok9",
    "270082434+crayfish-ai@users.noreply.github.com": "crayfish-ai",
    "241404605+MestreY0d4-Uninter@users.noreply.github.com": "MestreY0d4-Uninter",
    "268667990+Roy-oss1@users.noreply.github.com": "Roy-oss1",
    "27917469+nosleepcassette@users.noreply.github.com": "nosleepcassette",
    "241404605+MestreY0d4-Uninter@users.noreply.github.com": "MestreY0d4-Uninter",
    "109555139+davetist@users.noreply.github.com": "davetist",
    "39405770+yyq4193@users.noreply.github.com": "yyq4193",
    "Asunfly@users.noreply.github.com": "Asunfly",
    "2500400+honghua@users.noreply.github.com": "honghua",
    "462836+jplew@users.noreply.github.com": "jplew",
    "nish3451@users.noreply.github.com": "nish3451",
    "Mibayy@users.noreply.github.com": "Mibayy",
    "mibayy@users.noreply.github.com": "Mibayy",
    "135070653+sgaofen@users.noreply.github.com": "sgaofen",
    "lzy.dev@gmail.com": "zhiyanliu",
    "me@janstepanovsky.cz": "hhhonzik",
    "139848623+hhuang91@users.noreply.github.com": "hhuang91",
    "s.ozaki@ebinou.net": "Satoshi-agi",
    "10774721+kunlabs@users.noreply.github.com": "kunlabs",
    "110560187+Wang-tianhao@users.noreply.github.com": "Wang-tianhao",
    "170458616+ghostmfr@users.noreply.github.com": "ghostmfr",
    "1848670+mewwts@users.noreply.github.com": "mewwts",
    "1930707+haru398801@users.noreply.github.com": "haru398801",
    "rapabelias@gmail.com": "badgerbees",
    "xnb888@proton.me": "xnbi",
    "xiahu889889@proton.me": "xiahu88988",
    "nocoo@users.noreply.github.com": "nocoo",
    "30841158+n-WN@users.noreply.github.com": "n-WN",
    "tsuijinglei@gmail.com": "hiddenpuppy",
    "buraysandro9@gmail.com": "ygd58",
    "jerome@clawwork.ai": "HiddenPuppy",
    "jerome.benoit@sap.com": "jerome-benoit",
    "wysie@users.noreply.github.com": "Wysie",
    "leoyuan0099@gmail.com": "keyuyuan",
    "bxzt2006@163.com": "Only-Code-A",
    "i@troy-y.org": "TroyMitchell911",
    "mygamez@163.com": "zhongyueming1121",
    "hansnow@users.noreply.github.com": "hansnow",
    "134848055+UNLINEARITY@users.noreply.github.com": "UNLINEARITY",
    "ben.burtenshaw@gmail.com": "burtenshaw",
    "roopaknijhara@gmail.com": "rnijhara",
    "josephzcan@gmail.com": "j0sephz",
    # contributors (manual mapping from git names)
    "ahmedsherif95@gmail.com": "asheriif",
    "dyxushuai@gmail.com": "dyxushuai",
    "33860762+etcircle@users.noreply.github.com": "etcircle",
    "liujinkun@bytedance.com": "liujinkun2025",
    "dmayhem93@gmail.com": "dmahan93",
    "fr@tecompanytea.com": "ifrederico",
    "cdanis@gmail.com": "cdanis",
    "samherring99@gmail.com": "samherring99",
    "desaiaum08@gmail.com": "Aum08Desai",
    "shannon.sands.1979@gmail.com": "shannonsands",
    "shannon@nousresearch.com": "shannonsands",
    "abdi.moya@gmail.com": "AxDSan",
    "eri@plasticlabs.ai": "Erosika",
    "hjcpuro@gmail.com": "hjc-puro",
    "xaydinoktay@gmail.com": "aydnOktay",
    "abdullahfarukozden@gmail.com": "Farukest",
    "lovre.pesut@gmail.com": "rovle",
    "xjtumj@gmail.com": "mengjian-github",
    "kevinskysunny@gmail.com": "kevinskysunny",
    "xiewenxuan462@gmail.com": "yule975",
    "yiweimeng.dlut@hotmail.com": "meng93",
    "hakanerten02@hotmail.com": "teyrebaz33",
    "linux2010@users.noreply.github.com": "Linux2010",
    "elmatadorgh@users.noreply.github.com": "elmatadorgh",
    "coktinbaran5@gmail.com": "elmatadorgh",
    "alexazzjjtt@163.com": "alexzhu0",
    "1180176+Swift42@users.noreply.github.com": "Swift42",
    "ruzzgarcn@gmail.com": "Ruzzgar",
    "yukipukikedy@gmail.com": "Yukipukii1",
    "alireza78.crypto@gmail.com": "alireza78a",
    "brooklyn.bb.nicholson@gmail.com": "brooklynnicholson",
    "withapurpose37@gmail.com": "StefanIsMe",
    "4317663+helix4u@users.noreply.github.com": "helix4u",
    "ifkellx@users.noreply.github.com": "Ifkellx",
    "331214+counterposition@users.noreply.github.com": "counterposition",
    "blspear@gmail.com": "BrennerSpear",
    "akhater@gmail.com": "akhater",
    "Cos_Admin@PTG-COS.lodluvup4uaudnm3ycd14giyug.xx.internal.cloudapp.net": "akhater",
    "239876380+handsdiff@users.noreply.github.com": "handsdiff",
    "hesapacicam112@gmail.com": "etherman-os",
    "mark.ramsell@rivermounts.com": "mark-ramsell",
    "taeng02@icloud.com": "taeng0204",
    "gpickett00@gmail.com": "gpickett00",
    "mcosma@gmail.com": "wakamex",
    "clawdia.nash@proton.me": "clawdia-nash",
    "pickett.austin@gmail.com": "austinpickett",
    "dangtc94@gmail.com": "dieutx",
    "jaisehgal11299@gmail.com": "jaisup",
    "percydikec@gmail.com": "PercyDikec",
    "noonou7@gmail.com": "HenkDz",
    # Azure Foundry salvage (PRs #9029, #4599, #10086, #8766)
    "tech@smartlogics.net": "TechPrototyper",
    "637186+HangGlidersRule@users.noreply.github.com": "HangGlidersRule",
    "pein892@gmail.com": "pein892",
    "dean.kerr@gmail.com": "deankerr",
    "socrates1024@gmail.com": "socrates1024",
    "seanalt555@gmail.com": "Salt-555",
    "satelerd@gmail.com": "satelerd",
    "dan@danlynn.com": "danklynn",
    "mattmaximo@hotmail.com": "MattMaximo",
    "MatthewRHardwick@gmail.com": "mrhwick",
    "149063006+j3ffffff@users.noreply.github.com": "j3ffffff",
    "A-FdL-Prog@users.noreply.github.com": "A-FdL-Prog",
    "l0hde@users.noreply.github.com": "l0hde",
    "difujia@users.noreply.github.com": "difujia",
    "vominh1919@gmail.com": "vominh1919",
    "yue.gu2023@gmail.com": "YueLich",
    "51783311+andyylin@users.noreply.github.com": "andyylin",
    "me@jakubkrcmar.cz": "jakubkrcmar",
    "prasadus92@gmail.com": "prasadus92",
    "michael@make.software": "mssteuer",
    "der@konsi.org": "konsisumer",
    "abogale2@gmail.com": "amanuel2",
    "alexazzjjtt@163.com": "alexzhu0",
    "pub_forgreatagent@antgroup.com": "AntAISecurityLab",
    "252620095+briandevans@users.noreply.github.com": "briandevans",
    "danielrpike9@gmail.com": "Bartok9",
    "skozyuk@cruxexperts.com": "CruxExperts",
    "154585401+LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "12250313+Kailigithub@users.noreply.github.com": "Kailigithub",
    "mgparkprint@gmail.com": "vlwkaos",
    "1317078257maroon@gmail.com": "Oxidane-bot",
    "tranquil_flow@protonmail.com": "Tranquil-Flow",
    "LyleLengyel@gmail.com": "mcndjxlefnd",
    "wangshengyang2004@163.com": "Wangshengyang2004",
    "hasan.ali13381@gmail.com": "H-Ali13381",
    "xienb@proton.me": "XieNBi",
    "139681654+maymuneth@users.noreply.github.com": "maymuneth",
    "zengwei@nightq.cn": "nightq",
    "1434494126@qq.com": "5park1e",
    "158153005+5park1e@users.noreply.github.com": "5park1e",
    "innocarpe@gmail.com": "innocarpe",
    "noreply@ked.com": "qike-ms",
    "andrekurait@gmail.com": "AndreKurait",
    "bsgdigital@users.noreply.github.com": "bsgdigital",
    "numman.ali@gmail.com": "nummanali",
    "rohithsaimidigudla@gmail.com": "whitehatjr1001",
    "0xNyk@users.noreply.github.com": "0xNyk",
    "0xnykcd@googlemail.com": "0xNyk",
    "buraysandro9@gmail.com": "buray",
    "contact@jomar.fr": "joshmartinelle",
    "camilo@tekelala.com": "tekelala",
    "vincentcharlebois@gmail.com": "vincentcharlebois",
    "aryan@synvoid.com": "aryansingh",
    "johnsonblake1@gmail.com": "voteblake",
    "hcn518@gmail.com": "pedh",
    "haileymarshall005@gmail.com": "haileymarshall",
    "bennet.yr.wang@gmail.com": "BennetYrWang",
    "greer.guthrie@gmail.com": "g-guthrie",
    "kennyx102@gmail.com": "bobashopcashier",
    "77253505+bobashopcashier@users.noreply.github.com": "bobashopcashier",
    "25355950+megastary@users.noreply.github.com": "megastary",  # PR #18325
    "shokatalishaikh95@gmail.com": "areu01or00",
    "bryan@intertwinesys.com": "bryanyoung",
    "christo.mitov@gmail.com": "christomitov",
    "hermes@nousresearch.com": "NousResearch",
    "reginaldasr@gmail.com": "ReginaldasR",
    "ntconguit@gmail.com": "0xharryriddle",
    "agent@wildcat.local": "ericnicolaides",
    "georgex8001@gmail.com": "georgex8001",
    "stefan@dimagents.ai": "dimitrovi",
    "hermes@noushq.ai": "benbarclay",
    "chinmingcock@gmail.com": "ChimingLiu",
    "allard.quek@singtel.com": "AllardQuek",
    "openclaw@sparklab.ai": "openclaw",
    "semihcvlk53@gmail.com": "Himess",
    "erenkar950@gmail.com": "erenkarakus",
    "adavyasharma@gmail.com": "adavyas",
    "acaayush1111@gmail.com": "aayushchaudhary",
    "jason@outland.art": "jasonoutland",
    "73175452+Magaav@users.noreply.github.com": "Magaav",
    "mrflu1918@proton.me": "SPANISHFLU",
    "morganemoss@gmai.com": "mormio",
    "kopjop926@gmail.com": "cesareth",
    "fuleinist@gmail.com": "fuleinist",
    "jack.47@gmail.com": "JackTheGit",
    "dalvidjr2022@gmail.com": "Jr-kenny",
    "m@statecraft.systems": "mbierling",
    "balyan.sid@gmail.com": "alt-glitch",
    "52913345+alt-glitch@users.noreply.github.com": "alt-glitch",
    "oluwadareab12@gmail.com": "oluwadareab12",
    "simon@simonmarcus.org": "simon-marcus",
    "xowiekk@gmail.com": "Xowiek",
    "1243352777@qq.com": "zons-zhaozhy",
    "e.silacandmr@gmail.com": "Es1la",
    "51599529+stephen0110@users.noreply.github.com": "stephen0110",
    "265632032+sonic-netizen@users.noreply.github.com": "sonic-netizen",
    "82531659+mwnickerson@users.noreply.github.com": "mwnickerson",
    "sandrohub013@gmail.com": "SandroHub013",
    "maciekczech@users.noreply.github.com": "maciekczech",
    "h3057183414@gmail.com": "CoreyNoDream",
    "franksong2702@gmail.com": "franksong2702",
    "673088860@qq.com": "ambition0802",
    "beibei1988@proton.me": "beibi9966",
    # ── bulk addition: 75 emails resolved via API, PR salvage bodies, noreply
    #    crossref, and GH contributor list matching (April 2026 audit) ──
    "1115117931@qq.com": "aaronagent",
    "1506751656@qq.com": "hqhq1025",
    "364939526@qq.com": "luyao618",
    "hgk324@gmail.com": "houziershi",
    "176644217+PStarH@users.noreply.github.com": "PStarH",
    "51058514+Sanjays2402@users.noreply.github.com": "Sanjays2402",
    "16577466+andy825@user.noreply.gitee.com": "Andy283",
    "906014227@qq.com": "bingo906",
    "aaronwong1999@icloud.com": "AaronWong1999",
    "agents@kylefrench.dev": "DeployFaith",
    "angelos@oikos.lan.home.malaiwah.com": "angelos",
    "aptx4561@gmail.com": "cokemine",
    "arilotter@gmail.com": "ethernet8023",
    "ben@nousresearch.com": "benbarclay",
    "birdiegyal@gmail.com": "yyovil",
    "boschi1997@gmail.com": "nicoloboschi",
    "chef.ya@gmail.com": "cherifya",
    "chlqhdtn98@gmail.com": "BongSuCHOI",
    "coffeemjj@gmail.com": "Cafexss",
    "dalianmao0107@gmail.com": "dalianmao000",
    "der@konsi.org": "konsisumer",
    "dgrieco@redhat.com": "DomGrieco",
    "dhicham.pro@gmail.com": "spideystreet",
    "dipp.who@gmail.com": "dippwho",
    "don.rhm@gmail.com": "donrhmexe",
    "dorukardahan@hotmail.com": "dorukardahan",
    "dsocolobsky@gmail.com": "dsocolobsky",
    "dylan.socolobsky@lambdaclass.com": "dsocolobsky",
    "ignacio.avecilla@lambdaclass.com": "IAvecilla",
    "duerzy@gmail.com": "duerzy",
    "emozilla@nousresearch.com": "emozilla",
    "fancydirty@gmail.com": "fancydirty",
    "farion1231@gmail.com": "farion1231",
    "floptopbot33@gmail.com": "flobo3",
    "fontana.pedro93@gmail.com": "pefontana",
    "francis.x.fitzpatrick@gmail.com": "fxfitz",
    "frank@helmschrott.de": "Helmi",
    "gaixg94@gmail.com": "gaixianggeng",
    "geoff.wellman@gmail.com": "geoffwellman",
    "han.shan@live.cn": "jamesarch",
    "haolong@microsoft.com": "LongOddCode",
    "hata1234@gmail.com": "hata1234",
    "hmbown@gmail.com": "Hmbown",
    "iacobs@m0n5t3r.info": "m0n5t3r",
    "jiayuw794@gmail.com": "JiayuuWang",
    "jonny@nousresearch.com": "jquesnelle",
    "juan.ovalle@mistral.ai": "jjovalle99",
    "julien.talbot@ergonomia.re": "Julientalbot",
    "kagura.chen28@gmail.com": "kagura-agent",
    "1342088860@qq.com": "youngDoo",
    "kamil@gwozdz.me": "kamil-gwozdz",
    "skmishra1991@gmail.com": "bugkill3r",
    "karamusti912@gmail.com": "MustafaKara7",
    "kira@ariaki.me": "kira-ariaki",
    "knopki@duck.com": "knopki",
    "limars874@gmail.com": "limars874",
    "lisicheng168@gmail.com": "lesterli",
    "mingjwan@microsoft.com": "MagicRay1217",
    "orangeko@gmail.com": "GenKoKo",
    "82095453+iacker@users.noreply.github.com": "iacker",
    "sontianye@users.noreply.github.com": "sontianye",
    "jackjin1997@users.noreply.github.com": "jackjin1997",
    "1037461232@qq.com": "jackjin1997",
    "danieldoderlein@users.noreply.github.com": "danieldoderlein",
    "lrawnsley@users.noreply.github.com": "lrawnsley",
    "taeuk178@users.noreply.github.com": "taeuk178",
    "ogzerber@users.noreply.github.com": "ogzerber",
    "cola-runner@users.noreply.github.com": "cola-runner",
    "ygd58@users.noreply.github.com": "ygd58",
    "45554392+warabe1122@users.noreply.github.com": "warabe1122",
    "187001140+willy-scr@users.noreply.github.com": "willy-scr",
    "vominh1919@users.noreply.github.com": "vominh1919",
    "iamagenius00@users.noreply.github.com": "iamagenius00",
    "9219265+cresslank@users.noreply.github.com": "cresslank",
    "trevmanthony@gmail.com": "trevthefoolish",
    "ziliangpeng@users.noreply.github.com": "ziliangpeng",
    "centripetal-star@users.noreply.github.com": "centripetal-star",
    "LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "154585401+LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "cine.dreamer.one@gmail.com": "LeonSGP43",
    "Lubrsy706@users.noreply.github.com": "Lubrsy706",
    "niyant@spicefi.xyz": "spniyant",
    "olafthiele@gmail.com": "olafthiele",
    "oncuevtv@gmail.com": "sprmn24",
    "programming@olafthiele.com": "olafthiele",
    "r2668940489@gmail.com": "r266-tech",
    "s5460703@gmail.com": "BlackishGreen33",
    "saul.jj.wu@gmail.com": "SaulJWu",
    "shenhaocheng19990111@gmail.com": "hcshen0111",
    "sjtuwbh@gmail.com": "Cygra",
    "srhtsrht17@gmail.com": "Sertug17",
    "stephenschoettler@gmail.com": "stephenschoettler",
    "tanishq231003@gmail.com": "yyovil",
    "taosiyuan163@153.com": "taosiyuan163",
    "tesseracttars@gmail.com": "tesseracttars-creator",
    "tianliangjay@gmail.com": "xingkongliang",
    "1317078257maroon@gmail.com": "Oxidane-bot",
    "tranquil_flow@protonmail.com": "Tranquil-Flow",
    "LyleLengyel@gmail.com": "mcndjxlefnd",
    "unayung@gmail.com": "Unayung",
    "vorvul.danylo@gmail.com": "WorldInnovationsDepartment",
    "win4r@outlook.com": "win4r",
    "xush@xush.org": "KUSH42",
    "yangzhi.see@gmail.com": "SeeYangZhi",
    "yongtenglei@gmail.com": "yongtenglei",
    "young@YoungdeMacBook-Pro.local": "YoungYang963",
    "ysfalweshcan@gmail.com": "Junass1",
    "ysfwaxlycan@gmail.com": "WAXLYY",
    "yusufalweshdemir@gmail.com": "Dusk1e",
    "zhouboli@gmail.com": "zhouboli",
    "zqiao@microsoft.com": "tomqiaozc",
    "zzn+pa@zzn.im": "xinbenlv",
    "zaynjarvis@gmail.com": "ZaynJarvis",
    "zhiheng.liu@bytedance.com": "ZaynJarvis",
    "izhaolongfei@gmail.com": "loongfay",
    "296659110@qq.com": "lrt4836",
    "fe.daniel91@gmail.com": "beforeload",
    "libo1106@foxmail.com": "libo1106",
    "295367131@qq.com": "295367131",
    "295367132@qq.com": "IxAres",
    "danieldliu@tencent.com": "danieldliu",
    "loongzhao@tencent.com": "loongzhao",
    "Bartok9@users.noreply.github.com": "Bartok9",
    "LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "kshitijk4poor@users.noreply.github.com": "kshitijk4poor",
    "mbelleau@Michels-MacBook-Pro.local": "malaiwah",
    "michel.belleau@malaiwah.com": "malaiwah",
    "gnanasekaran.sekareee@gmail.com": "gnanam1990",
    "jz.pentest@gmail.com": "0xyg3n",
    "7093928+0xyg3n@users.noreply.github.com": "0xyg3n",
    "nftpoetrist@gmail.com": "nftpoetrist",  # PR #18982
    "millerc79@users.noreply.github.com": "millerc79",  # PR #19033
    "hermes@example.com": "shellybotmoyer",  # PR #18915 (bot-committed)
    "exx@example.com": "exxmen",  # PR #19555
    "hypnosis.mda@gmail.com": "Hypn0sis",
    "ywt000818@gmail.com": "OwenYWT",
    "dhandhalyabhavik@gmail.com": "v1k22",
    "rucchizhao@zhaochenfeideMacBook-Pro.local": "RucchiZ",
    "tannerfokkens@Mac.attlocal.net": "tannerfokkens-maker",
    "lehaolin98@outlook.com": "LehaoLin",
    "yuewang1@microsoft.com": "imink",
    "1736355688@qq.com": "hedgeho9X",
    "bernylinville@devopsthink.org": "bernylinville",
    "brian@bde.io": "briandevans",
    "hubin_ll@qq.com": "LLQWQ",
    "memosr_email@gmail.com": "memosr",
    "jperlow@gmail.com": "perlowja",
    "jasonpette1783@gmail.com": "web-dev0521",
    "bjianhang@gmail.com": "bjianhang",
    "tangyuanjc@JCdeAIfenshendeMac-mini.local": "tangyuanjc",
    "harryplusplus@gmail.com": "harryplusplus",
    "anthhub@163.com": "anthhub",
    "allard.quek@singtel.com": "AllardQuek",
    "shenuu@gmail.com": "shenuu",
    "xiayh17@gmail.com": "xiayh0107",
    "zhujianxyz@gmail.com": "opriz",
    "tuancanhnguyen706@gmail.com": "xxxigm",
    "asurla@nvidia.com": "anniesurla",
    "limkuan24@gmail.com": "WideLee",
    "aviralarora002@gmail.com": "AviArora02-commits",
    "draixagent@gmail.com": "draix",
    "junminliu@gmail.com": "JimLiu",
    "jarvischer@gmail.com": "maxchernin",
    "levantam.98.2324@gmail.com": "LVT382009",
    "zhurongcheng@rcrai.com": "heykb",
    "withapurpose37@gmail.com": "StefanIsMe",
    "261797239+lumenradley@users.noreply.github.com": "lumenradley",
    "166376523+sjz-ks@users.noreply.github.com": "sjz-ks",
    "haileymarshall005@gmail.com": "haileymarshall",
    "aniruddhaadak80@users.noreply.github.com": "aniruddhaadak80",
    "zheng.jerilyn@gmail.com": "jerilynzheng",
    "asslaenn5@gmail.com": "Aslaaen",
    "shalompmc0505@naver.com": "pinion05",
    "105142614+VTRiot@users.noreply.github.com": "VTRiot",
    "vivien000812@gmail.com": "iamagenius00",
    "89228157+Feranmi10@users.noreply.github.com": "Feranmi10",
    "oluwadareferanmi11@gmail.com": "Feranmi10",
    "simon@gtcl.us": "simon-gtcl",
    "suzukaze.haduki@gmail.com": "houko",
    "cliff@cigii.com": "cgarwood82",
    "anna@oa.ke": "anna-oake",
    "jaffarkeikei@gmail.com": "jaffarkeikei",
    "hxp@hxp.plus": "hxp-plus",
    "3580442280@qq.com": "Tianworld",
    "wujianxu91@gmail.com": "wujhsu",
    "zhrh120@gmail.com": "niyoh120",
    "vrinek@hey.com": "vrinek",
    "268198004+xandersbell@users.noreply.github.com": "xandersbell",
    "somme4096@gmail.com": "Somme4096",
    "brian@tiuxo.com": "brianclemens",
    "25944632+yudaiyan@users.noreply.github.com": "yudaiyan",
    "chayton@sina.com": "ycbai",
    "longsizhuo@gmail.com": "longsizhuo",
    "chenb19870707@gmail.com": "ms-alan",
    "276886827+WuTianyi123@users.noreply.github.com": "WuTianyi123",
    "22549957+li0near@users.noreply.github.com": "li0near",
    "23434080+sicnuyudidi@users.noreply.github.com": "sicnuyudidi",
    "haimu0x0@proton.me": "haimu0x",
    "abdelmajidnidnasser1@gmail.com": "NIDNASSER-Abdelmajid",
    "projectadmin@wit.id": "projectadmin-dev",
    "mrigankamondal10@gmail.com": "Dev-Mriganka",
    "132275809+shushuzn@users.noreply.github.com": "shushuzn",
    "ibrahimozsarac@gmail.com": "iborazzi",
    "130149563+A-afflatus@users.noreply.github.com": "A-afflatus",
    "huangkwell@163.com": "huangke19",
    "tanishq@exa.ai": "10ishq",
    "363708+christopherwoodall@users.noreply.github.com": "christopherwoodall",
    "zhang9w0v5@qq.com": "zhang9w0v5",
    "fuleinist@outlook.com": "fuleinist",
    "43494187+Llugaes@users.noreply.github.com": "Llugaes",
    "fengtianyu88@users.noreply.github.com": "fengtianyu88",
    "l.moncany@gmail.com": "lmoncany",
    "fatinghenji@users.noreply.github.com": "fatinghenji",
    "xin.peng.dr@gmail.com": "xinpengdr",
    "mike@mikewaters.net": "mikewaters",
    "65117428+WadydX@users.noreply.github.com": "WadydX",
    "216480837+isaachuangGMICLOUD@users.noreply.github.com": "isaachuangGMICLOUD",
    "isaac.h@gmicloud.ai": "isaachuangGMICLOUD",
    "nukuom976228@gmail.com": "hsy5571616",
    "11462216+Nan93@users.noreply.github.com": "Nan93",
    "l973401489@126.com": "zhouxiaoya12",
    "373119611@qq.com": "roytian1217",
    "brett@brettbrewer.com": "minorgod",
    "67779267+wenhao7@users.noreply.github.com": "wenhao7",
    "git@yzx9.xyz": "yzx9",
    "nilesh@cloudgeni.us": "lvnilesh",
    "63502660+azhengbot@users.noreply.github.com": "azhengbot",
    "sharvil.saxena@gmail.com": "sharziki",
    "yuanhe@minimaxi.com": "RyanLee-Dev",
    "curtis992250@gmail.com": "TaroballzChen",
    "92638503+Lind3ey@users.noreply.github.com": "Lind3ey",
    "1352808998@qq.com": "phpoh",
    "caliberoviv@gmail.com": "vivganes",
    "michaelfackerell@gmail.com": "MikeFac",
    "18024642@qq.com": "GuyCui",
    "eumael.mkt@gmail.com": "maelrx",
    # v0.11.0 additions
    "benbarclay@gmail.com": "benbarclay",
    "lijiawen@umich.edu": "Jiawen-lee",
    "oleksiy@kovyrin.net": "kovyrin",
    "kovyrin.claw@gmail.com": "kovyrin",
    "kaiobarb@gmail.com": "liftaris",
    "me@arihantsethia.com": "arihantsethia",
    "zhuofengwang2003@gmail.com": "coekfung",
    "teknium@noreply.github.com": "teknium1",
    "2114364329@qq.com": "cuyua9",
    "2557058999@qq.com": "Disaster-Terminator",
    "cine.dreamer.one@gmail.com": "LeonSGP43",
    "zyprothh@gmail.com": "Zyproth",
    "amitgaur@gmail.com": "amitgaur",
    "albuquerque.abner@gmail.com": "mrbob-git",
    "kiala@users.noreply.github.com": "kiala9",
    "alanxchen@gmail.com": "alanxchen85",
    "clawbot@clawbots-Mac-mini.local": "John-tip",
    "der@konsi.org": "konsisumer",
    "cirwel@The-CIRWEL-Group.local": "CIRWEL",
    "molvikar8@gmail.com": "molvikar",
    "nftpoetrist@gmail.com": "nftpoetrist",
    "dodofun@126.com": "colorcross",
    "1615063567@qq.com": "zhao0112",
    "ethanguo.2003@gmail.com": "EthanGuo-coder",
    "dev0jsh@gmail.com": "tmdgusya",
    "leavr@163.com": "leavrcn",
    "17683456+wanazhar@users.noreply.github.com": "wanazhar",
    "26782336+cixuuz@users.noreply.github.com": "cixuuz",
    "aleksandr.pasevin@openzeppelin.com": "pasevin",
    "ubuntu@localhost.localdomain": "holynn-q",
    "holynn@placeholder.local": "holynn-q",
    "agent@hermes.local": "jacdevos",
    "sunsky.lau@gmail.com": "liuhao1024",
    "qiuqfang98@qq.com": "keepcalmqqf",
    "261867348+ai-ag2026@users.noreply.github.com": "ai-ag2026",
    "yanzh.su@gmail.com": "YanzhongSu",
    "wanderwang@users.noreply.github.com": "WanderWang",
    "yueheime@gmail.com": "yuehei",
    "emidomh@gmail.com": "Emidomenge",
    "2642448440@qq.com": "BlackJulySnow",
    "4317663+helix4u@users.noreply.github.com": "helix4u",
    "floptopbot33@gmail.com": "flobo3",
    "dpaluy@users.noreply.github.com": "dpaluy",
    "psikonetik@gmail.com": "el-analista",
    "chenb19870707@gmail.com": "ms-alan",
    "hex-clawd@users.noreply.github.com": "hex-clawd",
    "154585401+LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "barteq@hacknotes.local": "barteqpl",
    "pama0227@gmail.com": "pama0227",
    "52785845+ee-blog@users.noreply.github.com": "ee-blog",
    "simplenamebox@gmail.com": "simplenamebox-ops",
    "balyan.sid@gmail.com": "alt-glitch",
    "xdord@xdorddeMac-mini.local": "foreverxdord",
    "k2767567815@gmail.com": "QifengKuang",
    "88077783+jjjojoj@users.noreply.github.com": "jjjojoj",
    "valda@underscore.jp": "valda",
    "lling486@163.com": "M3RCUR2Y",
    "buraysandro9@gmail.com": "ygd58",
    "ideathinklab01-source@users.noreply.github.com": "ideathinklab01-source",
    "27987889@qq.com": "zng8418",
    "daniuxie88@proton.me": "DaniuXie",
    "panchanler@gmail.com": "ChanlerDev",
    "252620095+briandevans@users.noreply.github.com": "briandevans",
    "141889580+h0tp-ftw@users.noreply.github.com": "h0tp-ftw",
    "chinadbo@foxmail.com": "chinadbo",
    "82637225+kshitijk4poor@users.noreply.github.com": "kshitijk4poor",
    "xyywtt@gmail.com": "xyiy001",
    "charliekerfoot@gmail.com": "CharlieKerfoot",
    "grey0202@users.noreply.github.com": "Grey0202",
    "vominh1919@gmail.com": "vominh1919",
    "giwavictor9@gmail.com": "giwaov",
    "yoimexex@gmail.com": "Yoimex",
    "76803960+atongrun@users.noreply.github.com": "atongrun",
    "michaeldanko@icloud.com": "MichaelWDanko",
    "xudavid429@gmail.com": "YX234",
    "kathy@Kathy.local": "julysir",
    "274902531@qq.com": "JanCong",
    "225304168+e-shizz@users.noreply.github.com": "e-shizz",
    "vincent_hh@users.noreply.github.com": "VinVC",
    "1243352777@qq.com": "zons-zhaozhy",
    "dejie.guo@gmail.com": "JayGwod",
    "52840391+swithek@users.noreply.github.com": "swithek",
    "raipratik0101@gmail.com": "PratikRai0101",
    "code@sasha.id": "sasha-id",
    "chen.yunbo@xydigit.com": "chenyunbo411",
    "openclaw@local": "Asce66",
    "59465365+0xsir0000@users.noreply.github.com": "0xsir0000",
    "lisanhu2014@hotmail.com": "lisanhu",
    "0668001438@zte.com.cn": "chenyunbo411",
    "steven_chanin@alum.mit.edu": "stevenchanin",
    "fiver@example.com": "halmisen",
    "mayq0422@gmail.com": "yuqianma",
    "yuqian@zmetasoft.com": "yuqianma",
    "scott@bubble.local": "bassings",
    "highland0971@users.noreply.github.com": "highland0971",
    "sudolewis@gmail.com": "lewislulu",
    "gaurav2301v@gmail.com": "Gaurav23V",
    "tranquil_flow@protonmail.com": "Tranquil-Flow",
    "albert748@gmail.com": "albert748",
    "ntconguit@gmail.com": "0xharryriddle",
    "lhysdl@gmail.com": "lhysdl",
    "shemol@163.com": "SherlockShemol",
    "clawdia@fmercurio-macstudio.local": "fmercurio",
    "ricardoporsche001@icloud.com": "Ricardo-M-L",
    "leozeli@qq.com": "leozeli",
    "linlehao@cuhk.edu.cn": "LehaoLin",
    "liutong@isacas.ac.cn": "I3eg1nner",
    "peterberthelsen@Peters-MacBook-Air.local": "PeterBerthelsen",
    "root@debian.debian": "lengxii",
    "roque@priveperfumeshn.com": "priveperfumes",
    "shijianzhi@shijianzhideMacBook-Pro.local": "sjz-ks",
    "topcheer@me.com": "topcheer",
    "walli@tencent.com": "walli",
    "zhuofengwang@tencent.com": "Zhuofeng-Wang",
    "simonweng@tencent.com": "Contentment003111",
    # April 2026 salvage-PR batch (#14920, #14986, #14966)
    "mrunmayeerane17@gmail.com": "mrunmayee17",
    "69489633+camaragon@users.noreply.github.com": "camaragon",
    "shamork@outlook.com": "shamork",
    # April 2026 Discord Copilot /model salvage (#15030)
    "cshong2017@outlook.com": "Nicecsh",
    # no-github-match — keep as display names
    "clio-agent@sisyphuslabs.ai": "Sisyphus",
    "marco@rutimka.de": "Marco Rutsch",
    "paul@gamma.app": "Paul Bergeron",
    "zhangxicen@example.com": "zhangxicen",
    "codex@openai.invalid": "teknium1",
    "screenmachine@gmail.com": "teknium1",
    "chenzeshi@live.com": "chen1749144759",
    "mor.aleksandr@yahoo.com": "MorAlekss",
    "276649498+ztexydt-cqh@users.noreply.github.com": "ztexydt-cqh",
    "ash@users.noreply.github.com": "ash",
    "andrewho.sf@gmail.com": "andrewhosf",
    # April 2026 Honcho bug-fix consolidation (#15381)
    "HiddenPuppy@users.noreply.github.com": "HiddenPuppy",
    "code@sasha.id": "sasha-id",
    "dontcallmejames@users.noreply.github.com": "dontcallmejames",
    "hekaru.agent@gmail.com": "hekaru-agent",
    "jas9000@gmail.com": "twozle",
    "r.filgueiras@apheris.com": "rfilgueiras",
    "leihaibo1992@gmail.com": "Leihb",
    # ACP streaming fix salvage (PR #9428 + #16273)
    "nfb0408@163.com": "ningfangbin",
    "164839249+Joseph19820124@users.noreply.github.com": "Joseph19820124",
    "rugved@lmstudio.ai": "rugvedS07",
    "44333070+Heltman@users.noreply.github.com": "Heltman",
    # v0.12.0 additions
    "ching@kachingappz.com": "ching-kaching",
    "codezhujr@gmail.com": "Zjianru",  # salvage chain: code by codez, PR #15749 author @Zjianru
    "daimon@noreply.github.com": "Siddharth Balyan",  # co-author only
    "i@zkl2333.com": "zkl2333",
    "isaachuang@Isaacs-MacBook-Pro.local": "isaachuangGMICLOUD",
    "isaachuang@Mac.localdomain": "isaachuangGMICLOUD",  # salvage of PR #11955 → #16663
    "liyuan851277048@icloud.com": "Octopus",  # co-author only
    "me+github7604@versun.org": "Versun",  # co-author only
    "my.vesper.nine@gmail.com": "kevin-ho",  # salvage: PR #15488 author @kevin-ho
    "noreply@paperclip.ing": "Paperclip",  # co-author only
    "teknium@hermes-agent": "teknium1",
    "web3blind@gmail.com": "web3blind",
    "ztzheng@163.com": "chengoak",  # PR #17467
    "24110240104@m.fudan.edu.cn": "YuShu",  # co-author only
    "charliekerfoot@gmail.com": "CharlieKerfoot",  # PR #18951
    # Debug share upload-time redaction (May 2026)
    "dhuysamen@gmail.com": "GodsBoy",  # PR #19318
    "mrcoferland@gmail.com": "mrcoferland",  # PR #19023
    "chenlinfeng@ruije.com.cn": "noOne-list",  # PR #19050
    "briansu@Mac-mini.attlocal.net": "likejudy",  # PR #19052
    "leosma@gmail.com": "leon7609",  # PR #19069
    "nouseman666@gmail.com": "nouseman666",  # PR #19088
    "ginwu05@gmail.com": "GinWU05",  # PR #19093
    "shashwatgokhe2@gmail.com": "shashwatgokhe",  # PR #19196
    "stevenchou.ai@gmail.com": "stevenchouai",  # PR #19221
    "leo.gong@phizchat.com": "agilejava",  # PR #19346
    "acc001k@pm.me": "acc001k",  # PR #19358
    "kowenhao@users.noreply.github.com": "kowenhaoai",  # PR #19376
    "hedirman@gmail.com": "hedirman",  # PR #19410
    "lucianopacheco@gmail.com": "LucianoSP",  # PR #19412
    "paultian.research@gmail.com": "paul-tian",  # PR #19423
    "info@glesperance.com": "glesperance",  # PR #19443
    "lxl694522264@gmail.com": "EvilDrag0n",  # PR #20651
    # v0.13.0 additions
    "clode@clo5de.info": "jackey8616",  # via PR salvage
    "james.russo@heygen.com": "jrusso1020",  # via PR salvage
    "leon@sgp43.com": "LeonSGP43",  # PR #18739 salvage of #14570
    "miniding@miniding.home": "Foolafroos",  # PR #20329 French locale
    "montbra@gmail.com": "Montbra",  # PR #20897 salvage of #16189 (TUI voice PTT)
    "promptsiren@gmail.com": "firefly",  # PR #18123 salvage of #16660 (ContextVars)
    "wtyopenclaw@gmail.com": "WuTianyi123",  # PR #20275 salvage of #13723 (feishu markdown)
    "zhicheng.han@mathematik.uni-goettingen.de": "hanzckernel",  # PR #20311 (api-server approval events)
    "agentsmithlaor@gmail.com": "oferlaor",  # PR #22356 salvage (cron origin sender identity)
    "jhin.lee@unity3d.com": "leehack",  # PR #22053 salvage (telegram DM topic reply fallback)
    # pander: empty email, salvaged via PR #19665 from #16126 by @ms-alan
    "ayman.a.kamal@hotmail.com": "A-kamal",  # PR #18678 (xAI image resolution fix)
}


def git(*args, cwd=None):
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True,
        cwd=cwd or str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed: {result.stderr}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def git_result(*args, cwd=None):
    """Run a git command and return the full CompletedProcess."""
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=cwd or str(REPO_ROOT),
    )


def get_last_tag():
    """Get the most recent CalVer tag."""
    tags = git("tag", "--list", "v20*", "--sort=-v:refname")
    if tags:
        return tags.split("\n")[0]
    return None


def next_available_tag(base_tag: str) -> tuple[str, str]:
    """Return a tag/calver pair, suffixing same-day releases when needed."""
    if not git("tag", "--list", base_tag):
        return base_tag, base_tag.removeprefix("v")

    suffix = 2
    while git("tag", "--list", f"{base_tag}.{suffix}"):
        suffix += 1
    tag_name = f"{base_tag}.{suffix}"
    return tag_name, tag_name.removeprefix("v")


def get_current_version():
    """Read current semver from __init__.py."""
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else "0.0.0"


def bump_version(current: str, part: str) -> str:
    """Bump a semver version string."""
    parts = current.split(".")
    if len(parts) != 3:
        parts = ["0", "0", "0"]
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump part: {part}")

    return f"{major}.{minor}.{patch}"


def update_version_files(semver: str, calver_date: str):
    """Update version strings in source files."""
    # Update __init__.py
    content = VERSION_FILE.read_text()
    content = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{semver}"',
        content,
    )
    content = re.sub(
        r'__release_date__\s*=\s*"[^"]+"',
        f'__release_date__ = "{calver_date}"',
        content,
    )
    VERSION_FILE.write_text(content)

    # Update pyproject.toml
    pyproject = PYPROJECT_FILE.read_text()
    pyproject = re.sub(
        r'^version\s*=\s*"[^"]+"',
        f'version = "{semver}"',
        pyproject,
        flags=re.MULTILINE,
    )
    PYPROJECT_FILE.write_text(pyproject)


def build_release_artifacts(semver: str) -> list[Path]:
    """Build sdist/wheel artifacts for the current release.

    Returns the artifact paths when the local environment has ``python -m build``
    available. If build tooling is missing or the build fails, returns an empty
    list and lets the release proceed without attached Python artifacts.
    """
    dist_dir = REPO_ROOT / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)

    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--wheel"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("  ⚠ Could not build Python release artifacts.")
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        if stderr:
            print(f"    {stderr.splitlines()[-1]}")
        elif stdout:
            print(f"    {stdout.splitlines()[-1]}")
        print("    Install the 'build' package to attach semver-named sdist/wheel assets.")
        return []

    artifacts = sorted(p for p in dist_dir.iterdir() if p.is_file())
    matching = [p for p in artifacts if semver in p.name]
    if not matching:
        print("  ⚠ Built artifacts did not match the expected release version.")
        return []
    return matching


def resolve_author(name: str, email: str) -> str:
    """Resolve a git author to a GitHub @mention."""
    # Try email lookup first
    gh_user = AUTHOR_MAP.get(email)
    if gh_user:
        return f"@{gh_user}"

    # Try noreply pattern
    noreply_match = re.match(r"(\d+)\+(.+)@users\.noreply\.github\.com", email)
    if noreply_match:
        return f"@{noreply_match.group(2)}"

    # Try username@users.noreply.github.com
    noreply_match2 = re.match(r"(.+)@users\.noreply\.github\.com", email)
    if noreply_match2:
        return f"@{noreply_match2.group(1)}"

    # Fallback to git name
    return name


def categorize_commit(subject: str) -> str:
    """Categorize a commit by its conventional commit prefix."""
    subject_lower = subject.lower()

    # Match conventional commit patterns
    patterns = {
        "breaking": [r"^breaking[\s:(]", r"^!:", r"BREAKING CHANGE"],
        "features": [r"^feat[\s:(]", r"^feature[\s:(]", r"^add[\s:(]"],
        "fixes": [r"^fix[\s:(]", r"^bugfix[\s:(]", r"^bug[\s:(]", r"^hotfix[\s:(]"],
        "improvements": [r"^improve[\s:(]", r"^perf[\s:(]", r"^enhance[\s:(]",
                         r"^refactor[\s:(]", r"^cleanup[\s:(]", r"^clean[\s:(]",
                         r"^update[\s:(]", r"^optimize[\s:(]"],
        "docs": [r"^doc[\s:(]", r"^docs[\s:(]"],
        "tests": [r"^test[\s:(]", r"^tests[\s:(]"],
        "chore": [r"^chore[\s:(]", r"^ci[\s:(]", r"^build[\s:(]",
                  r"^deps[\s:(]", r"^bump[\s:(]"],
    }

    for category, regexes in patterns.items():
        for regex in regexes:
            if re.match(regex, subject_lower):
                return category

    # Heuristic fallbacks
    if any(w in subject_lower for w in ["add ", "new ", "implement", "support "]):
        return "features"
    if any(w in subject_lower for w in ["fix ", "fixed ", "resolve", "patch "]):
        return "fixes"
    if any(w in subject_lower for w in ["refactor", "cleanup", "improve", "update "]):
        return "improvements"

    return "other"


def clean_subject(subject: str) -> str:
    """Clean up a commit subject for display."""
    # Remove conventional commit prefix
    cleaned = re.sub(r"^(feat|fix|docs|chore|refactor|test|perf|ci|build|improve|add|update|cleanup|hotfix|breaking|enhance|optimize|bugfix|bug|feature|tests|deps|bump)[\s:(!]+\s*", "", subject, flags=re.IGNORECASE)
    # Remove trailing issue refs that are redundant with PR links
    cleaned = cleaned.strip()
    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def parse_coauthors(body: str) -> list:
    """Extract Co-authored-by trailers from a commit message body.

    Returns a list of {'name': ..., 'email': ...} dicts.
    Filters out AI assistants and bots (Claude, Copilot, Cursor, etc.).
    """
    if not body:
        return []
    # AI/bot emails to ignore in co-author trailers
    _ignored_emails = {"noreply@anthropic.com", "noreply@github.com",
                       "cursoragent@cursor.com", "hermes@nousresearch.com"}
    _ignored_names = re.compile(r"^(Claude|Copilot|Cursor Agent|GitHub Actions?|dependabot|renovate)", re.IGNORECASE)
    pattern = re.compile(r"Co-authored-by:\s*(.+?)\s*<([^>]+)>", re.IGNORECASE)
    results = []
    for m in pattern.finditer(body):
        name, email = m.group(1).strip(), m.group(2).strip()
        if email in _ignored_emails or _ignored_names.match(name):
            continue
        results.append({"name": name, "email": email})
    return results


def get_commits(since_tag=None):
    """Get commits since a tag (or all commits if None)."""
    if since_tag:
        range_spec = f"{since_tag}..HEAD"
    else:
        range_spec = "HEAD"

    # Format: hash|author_name|author_email|subject\0body
    # Using %x00 (null) as separator between subject and body
    log = git(
        "log", range_spec,
        "--format=%H|%an|%ae|%s%x00%b%x00",
        "--no-merges",
    )

    if not log:
        return []

    commits = []
    # Split on double-null to get each commit entry, since body ends with \0
    # and format ends with \0, each record ends with \0\0 between entries
    for entry in log.split("\0\0"):
        entry = entry.strip()
        if not entry:
            continue
        # Split on first null to separate "hash|name|email|subject" from "body"
        if "\0" in entry:
            header, body = entry.split("\0", 1)
            body = body.strip()
        else:
            header = entry
            body = ""
        parts = header.split("|", 3)
        if len(parts) != 4:
            continue
        sha, name, email, subject = parts
        coauthor_info = parse_coauthors(body)
        coauthors = [resolve_author(ca["name"], ca["email"]) for ca in coauthor_info]
        commits.append({
            "sha": sha,
            "short_sha": sha[:8],
            "author_name": name,
            "author_email": email,
            "subject": subject,
            "category": categorize_commit(subject),
            "github_author": resolve_author(name, email),
            "coauthors": coauthors,
        })

    return commits


def get_pr_number(subject: str) -> str:
    """Extract PR number from commit subject if present."""
    match = re.search(r"#(\d+)", subject)
    if match:
        return match.group(1)
    return None


def generate_changelog(commits, tag_name, semver, repo_url="https://github.com/NousResearch/hermes-agent",
                       prev_tag=None, first_release=False):
    """Generate markdown changelog from categorized commits."""
    lines = []

    # Header
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    lines.append(f"# Hermes Agent v{semver} ({tag_name})")
    lines.append("")
    lines.append(f"**Release Date:** {date_str}")
    lines.append("")

    if first_release:
        lines.append("> 🎉 **First official release!** This marks the beginning of regular weekly releases")
        lines.append("> for Hermes Agent. See below for everything included in this initial release.")
        lines.append("")

    # Group commits by category
    categories = defaultdict(list)
    all_authors = set()
    teknium_aliases = {"@teknium1"}

    for commit in commits:
        categories[commit["category"]].append(commit)
        author = commit["github_author"]
        if author not in teknium_aliases:
            all_authors.add(author)
        for coauthor in commit.get("coauthors", []):
            if coauthor not in teknium_aliases:
                all_authors.add(coauthor)

    # Category display order and emoji
    category_order = [
        ("breaking", "⚠️ Breaking Changes"),
        ("features", "✨ Features"),
        ("improvements", "🔧 Improvements"),
        ("fixes", "🐛 Bug Fixes"),
        ("docs", "📚 Documentation"),
        ("tests", "🧪 Tests"),
        ("chore", "🏗️ Infrastructure"),
        ("other", "📦 Other Changes"),
    ]

    for cat_key, cat_title in category_order:
        cat_commits = categories.get(cat_key, [])
        if not cat_commits:
            continue

        lines.append(f"## {cat_title}")
        lines.append("")

        for commit in cat_commits:
            subject = clean_subject(commit["subject"])
            pr_num = get_pr_number(commit["subject"])
            author = commit["github_author"]

            # Build the line
            parts = [f"- {subject}"]
            if pr_num:
                parts.append(f"([#{pr_num}]({repo_url}/pull/{pr_num}))")
            else:
                parts.append(f"([`{commit['short_sha']}`]({repo_url}/commit/{commit['sha']}))")

            if author not in teknium_aliases:
                parts.append(f"— {author}")

            lines.append(" ".join(parts))

        lines.append("")

    # Contributors section
    if all_authors:
        # Sort contributors by commit count
        author_counts = defaultdict(int)
        for commit in commits:
            author = commit["github_author"]
            if author not in teknium_aliases:
                author_counts[author] += 1
            for coauthor in commit.get("coauthors", []):
                if coauthor not in teknium_aliases:
                    author_counts[coauthor] += 1

        sorted_authors = sorted(author_counts.items(), key=lambda x: -x[1])

        lines.append("## 👥 Contributors")
        lines.append("")
        lines.append("Thank you to everyone who contributed to this release!")
        lines.append("")
        for author, count in sorted_authors:
            commit_word = "commit" if count == 1 else "commits"
            lines.append(f"- {author} ({count} {commit_word})")
        lines.append("")

    # Full changelog link
    if prev_tag:
        lines.append(f"**Full Changelog**: [{prev_tag}...{tag_name}]({repo_url}/compare/{prev_tag}...{tag_name})")
    else:
        lines.append(f"**Full Changelog**: [{tag_name}]({repo_url}/commits/{tag_name})")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent Release Tool")
    parser.add_argument("--bump", choices=["major", "minor", "patch"],
                        help="Which semver component to bump")
    parser.add_argument("--publish", action="store_true",
                        help="Actually create the tag and GitHub release (otherwise dry run)")
    parser.add_argument("--date", type=str,
                        help="Override CalVer date (format: YYYY.M.D)")
    parser.add_argument("--first-release", action="store_true",
                        help="Mark as first release (no previous tag expected)")
    parser.add_argument("--output", type=str,
                        help="Write changelog to file instead of stdout")
    args = parser.parse_args()

    # Determine CalVer date
    if args.date:
        calver_date = args.date
    else:
        now = datetime.now()
        calver_date = f"{now.year}.{now.month}.{now.day}"

    base_tag = f"v{calver_date}"
    tag_name, calver_date = next_available_tag(base_tag)
    if tag_name != base_tag:
        print(f"Note: Tag {base_tag} already exists, using {tag_name}")

    # Determine semver
    current_version = get_current_version()
    if args.bump:
        new_version = bump_version(current_version, args.bump)
    else:
        new_version = current_version

    # Get previous tag
    prev_tag = get_last_tag()
    if not prev_tag and not args.first_release:
        print("No previous tags found. Use --first-release for the initial release.")
        print(f"Would create tag: {tag_name}")
        print(f"Would set version: {new_version}")

    # Get commits
    commits = get_commits(since_tag=prev_tag)
    if not commits:
        print("No new commits since last tag.")
        if not args.first_release:
            return

    print(f"{'='*60}")
    print(f"  Hermes Agent Release Preview")
    print(f"{'='*60}")
    print(f"  CalVer tag:      {tag_name}")
    print(f"  SemVer:          v{current_version} → v{new_version}")
    print(f"  Previous tag:    {prev_tag or '(none — first release)'}")
    print(f"  Commits:         {len(commits)}")
    print(f"  Unique authors:  {len(set(c['github_author'] for c in commits))}")
    print(f"  Mode:            {'PUBLISH' if args.publish else 'DRY RUN'}")
    print(f"{'='*60}")
    print()

    # Generate changelog
    changelog = generate_changelog(
        commits, tag_name, new_version,
        prev_tag=prev_tag,
        first_release=args.first_release,
    )

    if args.output:
        Path(args.output).write_text(changelog, encoding="utf-8")
        print(f"Changelog written to {args.output}")
    else:
        print(changelog)

    if args.publish:
        print(f"\n{'='*60}")
        print("  Publishing release...")
        print(f"{'='*60}")

        # Update version files
        if args.bump:
            update_version_files(new_version, calver_date)
            print(f"  ✓ Updated version files to v{new_version} ({calver_date})")

            # Commit version bump
            add_result = git_result("add", str(VERSION_FILE), str(PYPROJECT_FILE))
            if add_result.returncode != 0:
                print(f"  ✗ Failed to stage version files: {add_result.stderr.strip()}")
                return

            commit_result = git_result(
                "commit", "-m", f"chore: bump version to v{new_version} ({calver_date})"
            )
            if commit_result.returncode != 0:
                print(f"  ✗ Failed to commit version bump: {commit_result.stderr.strip()}")
                return
            print(f"  ✓ Committed version bump")

        # Create annotated tag
        tag_result = git_result(
            "tag", "-a", tag_name, "-m",
            f"Hermes Agent v{new_version} ({calver_date})\n\nWeekly release"
        )
        if tag_result.returncode != 0:
            print(f"  ✗ Failed to create tag {tag_name}: {tag_result.stderr.strip()}")
            return
        print(f"  ✓ Created tag {tag_name}")

        # Push
        push_result = git_result("push", "origin", "HEAD", "--tags")
        if push_result.returncode == 0:
            print(f"  ✓ Pushed to origin")
        else:
            print(f"  ✗ Failed to push to origin: {push_result.stderr.strip()}")
            print("    Continue manually after fixing access:")
            print("    git push origin HEAD --tags")

        # Build semver-named Python artifacts so downstream packagers
        # (e.g. Homebrew) can target them without relying on CalVer tag names.
        artifacts = build_release_artifacts(new_version)
        if artifacts:
            print("  ✓ Built release artifacts:")
            for artifact in artifacts:
                print(f"    - {artifact.relative_to(REPO_ROOT)}")

        # Create GitHub release
        changelog_file = REPO_ROOT / ".release_notes.md"
        changelog_file.write_text(changelog)

        gh_cmd = [
            "gh", "release", "create", tag_name,
            "--title", f"Hermes Agent v{new_version} ({calver_date})",
            "--notes-file", str(changelog_file),
        ]
        gh_cmd.extend(str(path) for path in artifacts)

        gh_bin = shutil.which("gh")
        if gh_bin:
            result = subprocess.run(
                gh_cmd,
                capture_output=True, text=True,
                cwd=str(REPO_ROOT),
            )
        else:
            result = None

        if result and result.returncode == 0:
            changelog_file.unlink(missing_ok=True)
            print(f"  ✓ GitHub release created: {result.stdout.strip()}")
            print(f"\n  🎉 Release v{new_version} ({tag_name}) published!")
        else:
            if result is None:
                print("  ✗ GitHub release skipped: `gh` CLI not found.")
            else:
                print(f"  ✗ GitHub release failed: {result.stderr.strip()}")
            print(f"    Release notes kept at: {changelog_file}")
            print(f"    Tag was created locally. Create the release manually:")
            print(
                f"    gh release create {tag_name} --title 'Hermes Agent v{new_version} ({calver_date})' "
                f"--notes-file .release_notes.md {' '.join(str(path) for path in artifacts)}"
            )
            print(f"\n  ✓ Release artifacts prepared for manual publish: v{new_version} ({tag_name})")
    else:
        print(f"\n{'='*60}")
        print(f"  Dry run complete. To publish, add --publish")
        print(f"  Example: python scripts/release.py --bump minor --publish")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
