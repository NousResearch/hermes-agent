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
import json
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

# ACP Registry manifest must stay version-locked with pyproject.toml.
# tests/acp/test_registry_manifest.py enforces this lockstep so the release
# bump touches both files atomically.
ACP_REGISTRY_MANIFEST = REPO_ROOT / "acp_registry" / "agent.json"

# ──────────────────────────────────────────────────────────────────────
# Git email → GitHub username mapping
# ──────────────────────────────────────────────────────────────────────

# Auto-extracted from noreply emails + manual overrides
AUTHOR_MAP = {
    "victor@rocketfueldev.com": "victor-kyriazakos",
    "87440198+JoaoMarcos44@users.noreply.github.com": "JoaoMarcos44",
    "286497132+srojk34@users.noreply.github.com": "srojk34",
    "59806492+sitkarev@users.noreply.github.com": "sitkarev",
    "zheng@omegasys.eu": "omegazheng",
    "220877172+james47kjv@users.noreply.github.com": "james47kjv",
    "yuhanglin@YuhangdeMac-mini.local": "1960697431",
    "admin@fent.quest": "XVVH",
    "despitemeguru@gmail.com": "definitelynotguru",
    "chaslui@outlook.com": "ChasLui",
    "rio.jeong@thebytesize.ai": "rio-jeong",
    "yehaotian@xuanshudeMac-mini.local": "ArcanePivot",
    "dbeyer7@gmail.com": "benegessarit",
    "264773240+MrDiamondBallz@users.noreply.github.com": "MrDiamondBallz",
    "94890352+Adolanium@users.noreply.github.com": "Adolanium",
    "kenmege@yahoo.com": "Kenmege",
    "tianying.x@eukarya.io": "xtymac",
    "dkobi16@gmail.com": "Diyoncrz18",
    "arnaud@nolimitdevelopment.com": "ali-nld",
    "sswdarius@gmail.com": "necoweb3",
    "peterhao@Peters-MacBook-Air.local": "pinguarmy",
    "joe.rinaldijohnson@shopify.com": "joerj123",
    "adalsteinnhelgason@Aalsteinns-MacBook-Pro-3.local": "AIalliAI",
    "adalsteinnhelgason@users.noreply.github.com": "AIalliAI",
    "iamlukethedev@users.noreply.github.com": "iamlukethedev",
    "zhang.hz6666@gmail.com": "HaozheZhang6",
    "barronlroth@gmail.com": "barronlroth",
    "ondrej.drapalik@gmail.com": "OndrejDrapalik",
    "tomasz.panek@gmail.com": "tomekpanek",
    "philipadsouza@gmail.com": "PhilipAD",
    "zhuhaoyu0909@icloud.com": "underthestars-zhy",
    "raysun12142006@gmail.com": "yanxue06",
    "alberto.regalado@ymail.com": "ARegalado1",
    "alchemistchaos@protonmail.com": "AlchemistChaos",  # co-author only
    "gilad@smiti.ai": "giladbau",
    "yusufalweshdemir@gmail.com": "Dusk1e",
    "804436395@qq.com": "LaPhilosophie",
    "maxmitcham@mac.home": "maxtrigify",
    "ccook@nvms.com": "ccook1963",
    "kristian@agrointel.no": "kristianvast",
    "thomas.paquette@gmail.com": "RyTsYdUp",
    "techxacm@gmail.com": "ProgramCaiCai",
    "266365592+bmoore210@users.noreply.github.com": "bmoore210",
    "123150002+deaneeth@users.noreply.github.com": "deaneeth",
    "157839748+psionic73@users.noreply.github.com": "psionic73",
    "manishbyatroy@gmail.com": "manishbyatroy",
    "chilltulpa@gmail.com": "TheGardenGallery",
    "al@randomsnowflake.me": "randomsnowflake",
    "zakame@zakame.net": "zakame",
    "152110621+jiangkoumo@users.noreply.github.com": "jiangkoumo",
    "qinhaojie.exe@bytedance.com": "qin-ctx",
    "834740219@qq.com": "ViewWay",
    "matt@vestigial.dev": "m4dni5",
    "harjoth.khara@gmail.com": "harjothkhara",
    "129007007+HeLLGURD@users.noreply.github.com": "HeLLGURD",
    "290859878+synapsesx@users.noreply.github.com": "synapsesx",
    "157689911+itsflownium@users.noreply.github.com": "itsflownium",
    "dirtyren@users.noreply.github.com": "dirtyren",
    "chanyoung.kim@nota.ai": "channkim",
    "stevenn.damatoo@gmail.com": "x1erra",
    "evansrory@gmail.com": "zimigit2020",
    "237263164+ft-ioxcs@users.noreply.github.com": "ft-ioxcs",
    "tharushkadinujaya05@gmail.com": "0xneobyte",
    "138671361+Veritas-7@users.noreply.github.com": "Veritas-7",
    "keiron@onehanded.com": "kmccammon",
    "268233388+CiarasClaws@users.noreply.github.com": "CiarasClaws",
    "amy@ravenwolf.de": "WolframRavenwolf",
    "github.com@wolfram.ravenwolf.de": "WolframRavenwolf",
    "895252509@qq.com": "895252509",
    "35259607+zxcasongs@users.noreply.github.com": "zxcasongs",
    "alfred@my-cloud.me": "alfred-smith-0",
    "tangtaizhong792@gmail.com": "tangtaizong666",
    "github@aldo.pw": "aldoeliacim",
    "max@c60spaceship.com": "MaxFreedomPollard",
    "achaljhawar03@gmail.com": "achaljhawar",
    "claytonchew@ClaytonMacMiniM4.local": "claytonchew",
    "hbentel@gmail.com": "hbentel",
    "JustinBao@outlook.com": "justinbao19",
    "kdunn926@gmail.com": "kdunn926",
    "mvanhorn@MacBook-Pro.local": "mvanhorn",
    "470766206@qq.com": "youjunxiaji",
    "mharris@parallel.ai": "NormallyGaussian",
    "roger@roger.local": "mollusk",
    "ted.malone@outlook.com": "temalo",
    "adityamalik2833@gmail.com": "alarcritty",
    "islam666@users.noreply.github.com": "islam666",
    "mnajafian@nvidia.com": "mnajafian-nv",
    "25539605+lsaether@users.noreply.github.com": "lsaether",
    "30080538+JimStenstrom@users.noreply.github.com": "JimStenstrom",
    "rod.boev@gmail.com": "rodboev",
    "70290504+dangelo352@users.noreply.github.com": "dangelo352",
    "zhaolei.vc@bytedance.com": "zhaoleibd",
    "jeffrobodie@gmail.com": "jeffrobodie-glitch",
    "kyssta-exe@users.noreply.github.com": "kyssta-exe",
    "ali.zakaee.1997@gmail.com": "ITheEqualizer",
    "copii.list@gmail.com": "stremtec",
    "solaiagent@gmail.com": "solaitken",
    "cryptoworlldz@gmail.com": "worlldz",
    "prostoandrei9@gmail.com": "vladkvlchk",
    "116314616+ThyFriendlyFox@users.noreply.github.com": "ThyFriendlyFox",
    "liliangjya@gmail.com": "truenorth-lj",
    "16943149+nepenth@users.noreply.github.com": "nepenth",
    "ben.bartholomew@vectorize.io": "benfrank241",
    "74339271+SaguaroDev@users.noreply.github.com": "SaguaroDev",
    "subw3@mail2.sysu.edu.cn": "Subway2023",
    "trevin@trevinchow.com": "tmchow",
    "zhipengli@thebrainly.ai": "a1245582339",
    "mathijs.vd.hurk@gmail.com": "mathijsvandenhurk",
    "david.gutowsky@gmail.com": "davidgut1982",
    "drpelagik@gmail.com": "SeaXen",
    "lengr@users.noreply.github.com": "LengR",
    "Kewe63@users.noreply.github.com": "Kewe63",
    "kewe.3217@gmail.com": "Kewe63",
    "17255546+CharZhou@users.noreply.github.com": "CharZhou",
    "metalclaudbot@gmail.com": "HashClawAI",
    "tonybear55665566@gmail.com": "TonyPepeBear",
    "kaspersniels@gmail.com": "nielskaspers",
    "daxxpasquini@gmail.com": "bpasquini",
    "kurobaryo@gmail.com": "kurobaryo",
    "scubamount@users.noreply.github.com": "scubamount",
    "251514042+youngstar-eth@users.noreply.github.com": "youngstar-eth",
    "155192176+alelpoan@users.noreply.github.com": "alelpoan",
    "aman@abacus.ai": "Aman113114-IITD",
    "octavio.turra@gmail.com": "octavioturra",
    "524706+Twanislas@users.noreply.github.com": "Twanislas",
    "9592417+adam91holt@users.noreply.github.com": "adam91holt",
    "kchuang1015@users.noreply.github.com": "kchuang1015",
    "maheshthedev@gmail.com": "MaheshtheDev",
    "kyssta-exe@users.noreply.github.com": "kyssta-exe",
    "shriganesh.patel@gmail.com": "ashishpatel26",
    "45688690+fujinice@users.noreply.github.com": "fujinice",
    "276689385+carltonawong@users.noreply.github.com": "carltonawong",
    "195255660+EvilHumphrey@users.noreply.github.com": "EvilHumphrey",
    "270604154+superearn-fisher@users.noreply.github.com": "superearn-fisher",
    "3540493+kpadilha@users.noreply.github.com": "kpadilha",
    "40378218+chaconne67@users.noreply.github.com": "chaconne67",
    "Pluviobyte@users.noreply.github.com": "Pluviobyte",
    "sanghyuk_seo@nexcubecorp.com": "sanghyuk-seo-nexcube",
    "subrtt@gmail.com": "Brixyy",
    "wangpuv@hotmail.com": "wangpuv",
    "202622897+ticketclosed-wontfix@users.noreply.github.com": "ticketclosed-wontfix",
    "wuxuebin1993@gmail.com": "victorGPT",
    "xiaoxingitee@gmail.com": "xiaoxinova",
    "wei.chen.coder@gmail.com": "wenchengxucool",
    "frowte3k@gmail.com": "Frowtek",
    "211828103+julio-cloudvisor@users.noreply.github.com": "julio-cloudvisor",
    "17778+kweiner@users.noreply.github.com": "kweiner",
    "223516181+faisfamilytravel@users.noreply.github.com": "faisfamilytravel",
    "45189813+baofuen@users.noreply.github.com": "baofuen",
    "interstellar.consulting@gmail.com": "Interstellar-code",
    "33978413+Interstellar-code@users.noreply.github.com": "Interstellar-code",
    "tillfalko@gmail.com": "tillfalko",
    "hi@fesalfayed.com": "fesalfayed",
    "marek.les@seznam.cz": "maxcz79",
    # teknium (multiple emails)
    "teknium1@gmail.com": "teknium1",
    "kenyon1977@gmail.com": "kenyonxu",
    "cipherframe@users.noreply.github.com": "CipherFrame",
    "donovan-yohan@users.noreply.github.com": "donovan-yohan",
    "121752779+jacevys@users.noreply.github.com": "jacevys",
    "me@promplate.dev": "CNSeniorious000",
    "yichengqiao21@gmail.com": "YarrowQiao",
    "erhanyasarx@gmail.com": "erhnysr",
    "draihan@student.ubc.ca": "0xdany",  # PR #26124 salvage (chat argv off event loop)
    "30366221+WorldWriter@users.noreply.github.com": "WorldWriter",
    "dafeng@DafengdeMacBook-Pro.local": "WorldWriter",
    "schepers.zander1@gmail.com": "Strontvod",
    "ed@bebop.crew": "someaka",
    "anadi.jaggia@gmail.com": "Jaggia",
    "steve@steveonjava.com": "steveonjava",
    "steveonjava@gmail.com": "steveonjava",
    "squiddy@2rook.ai": "MoonRay305",
    "annguyenNous@users.noreply.github.com": "annguyenNous",
    "32201324+simpolism@users.noreply.github.com": "simpolism",
    "simpolism@gmail.com": "simpolism",
    "jake@nousresearch.com": "simpolism",
    "mgongzai@gmail.com": "vKongv",
    "0x.badfriend@gmail.com": "discodirector",
    "altriatree@gmail.com": "TruaShamu",
    "contact-me@stark-x.cn": "Stark-X",
    "nat@nthrow.io": "nthrow",
    "m@mobrienv.dev": "mikeyobrien",
    "saeed919@pm.me": "falasi",
    "chrisdlc119@outlook.com": "chdlc",
    "omar@techdeveloper.site": "nycomar",
    "qiyin.zuo@pcitc.com": "qiyin-code",
    "mr.aashiz@gmail.com": "aashizpoudel",
    "adityargadgil@gmail.com": "AdityaRajeshGadgil",
    "70629228+shaun0927@users.noreply.github.com": "shaun0927",
    "soju06@users.noreply.github.com": "Soju06",
    "34199905+Soju06@users.noreply.github.com": "Soju06",
    "98262967+Bihruze@users.noreply.github.com": "Bihruze",
    "189280367+Lempkey@users.noreply.github.com": "Lempkey",
    "34853915+m0n3r0@users.noreply.github.com": "m0n3r0",
    "leeseoki@makestar.com": "leeseoki0",
    "kronexoi13@gmail.com": "kronexoi",
    "hua.zhong@kingsmith.com": "vgocoder",
    "hermes@marian.local": "Schrotti77",
    "david@memorilabs.ai": "devwdave",
    "dave@devwdave.com": "devwdave",
    "1920071390@campus.ouj.ac.jp": "zapabob",
    "zapabob@users.noreply.github.com": "zapabob",
    "gaia@gaia.local": "jfuenmayor",
    "jiahuigu@users.noreply.github.com": "Jiahui-Gu",
    "openhands@all-hands.dev": "YLChen-007",
    "3153586+xzessmedia@users.noreply.github.com": "xzessmedia",
    "AdamPlatin123@outlook.com": "AdamPlatin123",
    "32711803+waefrebeorn@users.noreply.github.com": "waefrebeorn",
    "32869278+dusterbloom@users.noreply.github.com": "dusterbloom",
    "189737461+basilalshukaili@users.noreply.github.com": "basilalshukaili",
    "liuhao1024@users.noreply.github.com": "liuhao1024",
    "Rivuza@users.noreply.github.com": "Rivuza",
    "annguyenNous@users.noreply.github.com": "annguyenNous",
    "285874597+annguyenNous@users.noreply.github.com": "annguyenNous",
    "kylekahraman@users.noreply.github.com": "kylekahraman",
    "130975919+kylekahraman@users.noreply.github.com": "kylekahraman",
    "seppe@fushia.be": "seppegadeyne",
    "18264851+seppegadeyne@users.noreply.github.com": "seppegadeyne",
    "blackpilledsoftware@gmail.com": "blackpilledsoftware-prog",
    "266800570+blackpilledsoftware-prog@users.noreply.github.com": "blackpilledsoftware-prog",
    "dsr-restyn@users.noreply.github.com": "dsr-restyn",
    "210765158+WuKongAI-CMU@users.noreply.github.com": "WuKongAI-CMU",
    "lichriszhang@gmail.com": "codeblackhole1024",
    "leovillalbajr@gmail.com": "Lempkey",
    "nidhi2894@gmail.com": "nidhi-singh02",
    "30312689+aashizpoudel@users.noreply.github.com": "aashizpoudel",
    "oleksii.lisikh@gmail.com": "olisikh",
    "jithendranaidunara@gmail.com": "JithendraNara",
    "islam666@users.noreply.github.com": "islam666",
    "30467832+islam666@users.noreply.github.com": "islam666",
    "jeremy@geocaching.com": "outdoorsea",
    "54763683+thedavidmurray@users.noreply.github.com": "thedavidmurray",
    "leone.parise@gmail.com": "leoneparise",
    "mr@shu.io": "mrshu",
    "adam.manning@gmail.com": "am423",
    "buraysandro9@gmail.com": "ygd58",
    "108427749+buntingszn@users.noreply.github.com": "buntingszn",
    "yanglongwei06@gmail.com": "Alex-yang00",
    "yanghongda@jackyun.com": "yangguangjin",
    "teknium@nousresearch.com": "teknium1",
    "markuscontasul@gmail.com": "Glucksberg",
    "80581902+Glucksberg@users.noreply.github.com": "Glucksberg",
    "piyushvp1@gmail.com": "thelumiereguy",
    "pnascimento9596@gmail.com": "pnascimento9596",
    "dskwelmcy@163.com": "dskwe",
    "421774554@qq.com": "wuli666",
    "twebefy@gmail.com": "tw2818",
    "harish.kukreja@gmail.com": "counterposition",
    "korkyzer@gmail.com": "Korkyzer",
    "1046611633@qq.com": "zhengyn0001",
    "1095245867@qq.com": "littlewwwhite",
    "db@project-aeon.com": "db-aeon",
    "ahmed@abadr.net": "ahmedbadr3",
    "63822243+CoinTheHat@users.noreply.github.com": "CoinTheHat",
    "cleo@edaphic.xyz": "curiouscleo",
    "hirokazu.ogawa@kwansei.ac.jp": "hrkzogw",
    "datapod.k@gmail.com": "dandacompany",
    "treydong.zh@gmail.com": "TreyDong",
    "phil.thomas@gametime.co": "explainanalyze",
    "kyanam.preetham@gmail.com": "pkyanam",
    "zhizhong.xu@shopee.com": "1000Delta",
    "30397170+1000Delta@users.noreply.github.com": "1000Delta",
    "szymonclawd@mac.home": "szymonclawd",
    "257759490+szymonclawd@users.noreply.github.com": "szymonclawd",
    "101180447+worlldz@users.noreply.github.com": "worlldz",
    "zhanganzhe@tenclass.com": "luoyuctl",
    "51604064+luoyuctl@users.noreply.github.com": "luoyuctl",
    "127238744+teknium1@users.noreply.github.com": "teknium1",
    "tolle.lege+github@gmail.com": "InB4DevOps",
    "73686890+InB4DevOps@users.noreply.github.com": "InB4DevOps",
    "147827411+EloquentBrush@users.noreply.github.com": "AhmetArif0",
    "97489706+purzbeats@users.noreply.github.com": "purzbeats",
    "hugosequier@gmail.com": "Hugo-SEQUIER",
    "kylejeong21@gmail.com": "Kylejeong2",
    "128259593+Gutslabs@users.noreply.github.com": "Gutslabs",
    "50326054+nocturnum91@users.noreply.github.com": "nocturnum91",
    "52470719+gianfrancopiana@users.noreply.github.com": "gianfrancopiana",
    "223003280+Abd0r@users.noreply.github.com": "Abd0r",
    "HuangYuChuh@users.noreply.github.com": "HuangYuChuh",
    "aaronwong1989@gmail.com": "hrygo",
    "26729613+hrygo@users.noreply.github.com": "hrygo",
    "erenkar950@gmail.com": "eren-karakus0",
    "aubrey@freeman-wisco.com": "Freeman-Consulting",
    "don.rhm@gmail.com": "rahimsais",
    "40222899+rahimsais@users.noreply.github.com": "rahimsais",
    "alfred@Alfreds-Mac-mini.local": "NivOO5",
    "231191380+NivOO5@users.noreply.github.com": "NivOO5",
    "jameshuang@gmail.com": "kjames2001",
    "62420081+kjames2001@users.noreply.github.com": "kjames2001",
    "132184373+wilsen0@users.noreply.github.com": "wilsen0",
    "ra2157218@gmail.com": "Abd0r",
    "oswaldb22@users.noreply.github.com": "oswaldb22",
    "abdielv@proton.me": "AJV20",
    "mason@growagainorchids.com": "masonjames",
    "108541149+amethystani@users.noreply.github.com": "amethystani",
    "ytchen0719@gmail.com": "liquidchen",
    "am@studio1.tailb672fe.ts.net": "subtract0",
    "mike@grossmann.at": "ReqX",
    "axmaiqiu@gmail.com": "qWaitCrypto",
    "44045911+kidonng@users.noreply.github.com": "kidonng",
    "daniellsmarta@gmail.com": "DanielLSM",
    "264291321+v1b3coder@users.noreply.github.com": "v1b3coder",
    "silverchris@foxmail.com": "ming1523",
    "maksesipov@gmail.com": "Qwinty",
    "byquenox@gmail.com": "Que0x",
    "denisamania@gmail.com": "CalmProton",
    "308068+mbac@users.noreply.github.com": "mbac",
    "nicoechaniz@altermundi.net": "nicoechaniz",
    "ninso112@proton.me": "Ninso112",
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
    "eloklam2002@gmail.com": "eloklam",
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
    "bartok9@users.noreply.github.com": "Bartok9",
    "erhanyasarx@gmail.com": "erhnysr",  # PR #25198 salvage (tool-progress flood-control)
    "cryptobyz.airdrop@gmail.com": "CryptoByz",  # PR #25630 salvage (polling conflict Stage 1+2)
    "fabioxxx@gmail.com": "fabiosiqueira",  # PR #27212 salvage (bg-process notif anchor)
    "lordfalcon.exe@gmail.com": "falconexe",  # PR #24511 salvage (sticky-IP reset)
    "fonhal@gmail.com": "fonhal",  # PR #27865/#27861 salvage (mention entities / typing fallback)
    "zyrixtrex@gmail.com": "Zyrixtrex",  # PR #26754 salvage (avoid duplicate text after auto-TTS)
    "264138787+nftpoetrist@users.noreply.github.com": "nftpoetrist",  # PR #25856 salvage (escape slash-confirm preview)
    "197455947+samahn0601@users.noreply.github.com": "samahn0601",  # PR #27887 salvage (retry wrapped connect timeouts)
    "gonzes7@gmail.com": "aqilaziz",  # PR #26406 salvage (preserve native audio outside Telegram)
    "karthikeyann@users.noreply.github.com": "karthikeyann",  # PR #26609 salvage (DM-topic routing pin)
    "rino.alpin@gmail.com": "kunci115",  # PR #27098 salvage (thread-not-found retry)
    "hayka-pacha@users.noreply.github.com": "hayka-pacha",  # PR #25270 salvage (registry-aware mcp_ prefix strip)
    "237601532+chromalinx@users.noreply.github.com": "chromalinx",  # PR #27014 salvage (commands for groups+DM)
    "chromalinx@users.noreply.github.com": "chromalinx",  # PR #37026 salvage (SSL CA guard)
    "booker1207@gmail.com": "booker1207",  # PR #25132 salvage (gate profile bots by allowed topics)
    "kiranvk2011@gmail.com": "kiranvk-2011",  # PR #24815 salvage (image documents → vision)
    "kosmonaut-t@centrum.cz": "rak135",  # PR #25960 salvage (Windows /restart)
    "bot.chi.online@gmail.com": "B0Tch1",  # PR #27634 salvage (disable_topic_auto_rename)
    "1037461232@qq.com": "jackjin1997",  # PR #27239 salvage (restore DM topic thread_id after split)
    "soynchuux@gmail.com": "soynchux",  # PR #27806 salvage (chat-scoped auth without user_id)
    "psikonetik@gmail.com": "el-analista",  # PR #25368 salvage (cron topic fallback report)
    "75435655+khungate@users.noreply.github.com": "khungate",  # PR #25829 salvage (gmail-triage gt: callbacks)
    "stevehq26-bot@users.noreply.github.com": "stevehq26-bot",  # PR #28015 salvage (quick-command-only menus)
    "seaverb@icloud.com": "brndnsvr",  # PR #25327 salvage (channel post updates)
    "oracle@jarviss-mbp.home": "houenyang-momo",  # PR #24014 salvage (quiet noisy errors)
    "57119977+OCWC22@users.noreply.github.com": "OCWC22",  # PR #24581 salvage (multi-bot exclusive mentions)
    "ai-hana-ai@users.noreply.github.com": "ai-hana-ai",  # PR #23928 salvage (ignore_root_dm)
    "mx.indigo.karasu@gmail.com": "indigokarasu",  # PR #26636 salvage (pin user message)
    "516972+alber70g@users.noreply.github.com": "alber70g",  # PR #25280 salvage (skip-STT + 2GB cap)
    "282919977+eliteworkstation94-ai@users.noreply.github.com": "eliteworkstation94-ai",  # PR #28157 salvage (group reply session splits)
    "androidhtml@yandex.com": "hllqkb",
    "25840394+Bongulielmi@users.noreply.github.com": "Bongulielmi",
    "jonathan.troyer@overmatch.com": "JTroyerOvermatch",
    "harryykyle1@gmail.com": "hharry11",
    "wysie@users.noreply.github.com": "wysie",
    "ronhi@buildabear1.localdomain": "RonHillDev",  # PR #29523 salvage (machine-local commit email)
    "moikapy@devmoi.com": "Moikapy",  # PR #31527 salvage
    "barany.gabor@gmail.com": "gbarany",  # PR #27907 salvage (xAI sanitizer deepcopy)
    "hello@nami4d.tech": "Nami4D",  # PR #28490 salvage
    "jkausel@gmail.com": "jkausel-ai",
    "e.silacandmr@gmail.com": "Es1la",
    "51599529+stephen0110@users.noreply.github.com": "stephen0110",
    "265632032+sonic-netizen@users.noreply.github.com": "sonic-netizen",
    "82531659+mwnickerson@users.noreply.github.com": "mwnickerson",
    "sandrohub013@gmail.com": "SandroHub013",
    "maciekczech@users.noreply.github.com": "maciekczech",
    "154585401+LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "cine.dreamer.one@gmail.com": "LeonSGP43",
    "david@nutricraft.ca": "cyb0rgk1tty",
    "214562553+cyb0rgk1tty@users.noreply.github.com": "cyb0rgk1tty",
    "11052595+chimpera@users.noreply.github.com": "chimpera",
    "chris+dora@cmullins.io": "cmullins70",
    "zjtan1@gmail.com": "zeejaytan",
    "asslaenn5@gmail.com": "Aslaaen",
    "trae.anderson17@icloud.com": "Tkander1715",
    "beardthelion@users.noreply.github.com": "beardthelion",
    "orkunozturk@gmail.com": "orcool",
    "tangyuanjc@JCdeAIfenshendeMac-mini.local": "tangyuanjc",
    "leon@agentlinker.ai": "agentlinker",
    "santoshhumagain1887@gmail.com": "npmisantosh",
    "39641663+luarss@users.noreply.github.com": "luarss",
    "16263913+zccyman@users.noreply.github.com": "zccyman",
    "zccyman@users.noreply.github.com": "zccyman",  # PR #26998 (auxiliary fallback chain)
    "ahmetosrak@Ahmet-MacBook-Air.local": "Osraka",
    "98612432+Osraka@users.noreply.github.com": "Osraka",
    "112634774+ryptotalent@users.noreply.github.com": "ryptotalent",
    "270097726+hookinglau@users.noreply.github.com": "hookinglau",
    "5029547+AllynSheep@users.noreply.github.com": "AllynSheep",
    "allyn0306@gmail.com": "AllynSheep",
    "46887634+aqilaziz@users.noreply.github.com": "aqilaziz",
    "gonzes7@gmail.com": "aqilaziz",
    "6966326+laoli-no1@users.noreply.github.com": "laoli-no1",
    "laoli_no1@163.com": "laoli-no1",
    "39730900+NorethSea@users.noreply.github.com": "NorethSea",
    "963979204@qq.com": "NorethSea",
    "2283389+JamesX88@users.noreply.github.com": "JamesX88",
    "JamesX88@users.noreply.github.com": "JamesX88",
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
    # LINE messaging plugin (synthesis PR)
    "32443648+leepoweii@users.noreply.github.com": "leepoweii",
    "openclaw@liyangchen.me": "liyoungc",
    "charles@perng.com": "perng",
    "soichiro0111.dev@gmail.com": "soichiyo",
    "0xde@pieverse.io": "David-0x221Eight",
    "77736378+David-0x221Eight@users.noreply.github.com": "David-0x221Eight",
    "74749461+yuga-hashimoto@users.noreply.github.com": "yuga-hashimoto",
    "xiangyong@zspace.cn": "CES4751",
    "harish.kukreja@gmail.com": "counterposition",
    "nidhi2894@gmail.com": "nidhi-singh02",
    "35294173+Fearvox@users.noreply.github.com": "Fearvox",
    "fearvox1015@gmail.com": "Fearvox",
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
    "14341805+pty819@users.noreply.github.com": "pty819",
    "517024110@qq.com": "chennest",
    # Curator fixes (Apr 30 2026)
    "yuxiangl490@gmail.com": "y0shua1ee",
    "manmit0x@gmail.com": "0xDevNinja",
    "stevekelly622@gmail.com": "steezkelly",
    "brian@dralth.com": "btorresgil",
    "momowind@gmail.com": "momowind",
    "clockwork-codex@users.noreply.github.com": "misery-hl",
    "207811921+misery-hl@users.noreply.github.com": "misery-hl",
    "20nik.nosov21@gmail.com": "nik1t7n",
    "90299797+nik1t7n@users.noreply.github.com": "nik1t7n",
    "suncokret@protonmail.com": "suncokret12",
    "WompaJango@protonmail.com": "WompaJango",
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
    # Codex Spark restoration salvage (May 2026)
    "olegwn@gmail.com": "nederev",
    "vesper@askclaw.dev": "askclaw-vesper",
    "nazirulhafiy@gmail.com": "nazirulhafiy",
    "CREWorx@users.noreply.github.com": "BadTechBandit",
    "yoimexex@gmail.com": "Yoimex",
    "6548898+romanornr@users.noreply.github.com": "romanornr",
    "foxion37@gmail.com": "foxion37",
    "bloodcarter@gmail.com": "bloodcarter",
    "scott@scotttrinh.com": "scotttrinh",
    "quocanh261997@gmail.com": "quocanh261997",
    "savanne.kham@protonmail.com": "savanne-kham",  # PR #28958 salvage (strip tool_name for strict providers)
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
    "123342691+banditburai@users.noreply.github.com": "banditburai",
    "9063726+Kyzcreig@users.noreply.github.com": "Kyzcreig",
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
    "mibay@clawhub.io": "Mibayy",
    "louismichalot@hotmail.com": "Mibayy",
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
    "brooklyn.bb.nicholson@gmail.com": "OutThisLife",
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
    "incharge.automation@gmail.com": "inchargeautomation-lab",
    "danielrpike9@gmail.com": "Bartok9",
    "96944678+ymylive@users.noreply.github.com": "sweetcornna",
    "laflamme@illinoisalumni.org": "briancl2",
    "skozyuk@cruxexperts.com": "CruxExperts",
    "154585401+LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "12250313+Kailigithub@users.noreply.github.com": "Kailigithub",
    "mgparkprint@gmail.com": "vlwkaos",
    "1317078257maroon@gmail.com": "Oxidane-bot",
    "tranquil_flow@protonmail.com": "Tranquil-Flow",
    "66773372+Tranquil-Flow@users.noreply.github.com": "Tranquil-Flow",
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
    "jack@jackyang.com": "0xjackyang",
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
    "1115117931@qq.com": "aaronlab",
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
    "glennc@microsoft.com": "glennc",
    "hata1234@gmail.com": "hata1234",
    "hmbown@gmail.com": "Hmbown",
    "iacobs@m0n5t3r.info": "m0n5t3r",
    "jiayuw794@gmail.com": "JiayuuWang",
    "jonny@nousresearch.com": "yoniebans",
    "jake@nousresearch.com": "simpolism",
    "juan.ovalle@mistral.ai": "jjovalle99",
    "julien.talbot@ergonomia.re": "Julientalbot",
    "kagura.chen28@gmail.com": "kagura-agent",
    "1342088860@qq.com": "youngDoo",
    "kamil@gwozdz.me": "kamil-gwozdz",
    "skmishra1991@gmail.com": "bugkill3r",
    "karamusti912@gmail.com": "MustafaKara7",
    "kira@ariaki.me": "kira-ariaki",
    "kira.ops@proton.me": "KiraKatana",
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
    "ziliangdotme@gmail.com": "ziliangpeng",
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
    "56406949+RaumfahrerSpiffy@users.noreply.github.com": "Spaceman-Spiffy",  # PR #35586 (renamed account)
    "ian@culling.ca": "ianculling",  # PR #36087
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
    "vmphuongit@gmail.com": "phuongvm",
    "allard.quek@singtel.com": "AllardQuek",
    "shenuu@gmail.com": "shenuu",
    "xiayh17@gmail.com": "xiayh0107",
    "zhujianxyz@gmail.com": "opriz",
    "tuancanhnguyen706@gmail.com": "xxxigm",
    "larcombe.n@gmail.com": "NickLarcombe",
    "54813621+xxxigm@users.noreply.github.com": "xxxigm",
    "asurla@nvidia.com": "anniesurla",
    "kchantharuan@nvidia.com": "nv-kasikritc",
    "bbednarski@nvidia.com": "bbednarski9",
    "limkuan24@gmail.com": "WideLee",
    "aviralarora002@gmail.com": "AviArora02-commits",
    "draixagent@gmail.com": "draix",
    "martin.alca@gmail.com": "draix",
    "junminliu@gmail.com": "JimLiu",
    "juraj@bednar.io": "jooray",
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
    "agorgianitisj@hotmail.com": "johnisag",
    "phil.thomas@gametime.co": "explainanalyze",
    "276886827+WuTianyi123@users.noreply.github.com": "WuTianyi123",
    "22549957+li0near@users.noreply.github.com": "li0near",
    "guoyu801@gmail.com": "li0near",
    "ty@tmrtn.com": "tymrtn",
    "elitovsky@zenproject.net": "kallidean",
    "5463986+baocin@users.noreply.github.com": "baocin",
    "107296821+princepal9120@users.noreply.github.com": "princepal9120",
    "gufo0125@gmail.com": "guglielmofonda",
    "102474490+yehuosi@users.noreply.github.com": "yehuosi",
    "yehuosi@users.noreply.github.com": "yehuosi",
    "31932854+jelrod27@users.noreply.github.com": "jelrod27",
    "11262660+konsisumer@users.noreply.github.com": "konsisumer",
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
    "xiangji.chen@centurygame.com": "Llugaes",
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
    "rob@rbrtbn.com": "rbrtbn",
    "haaasined@gmail.com": "VinciZhu",
    "fabianoeq@gmail.com": "rodrigoeqnit",
    "178342791+sgtworkman@users.noreply.github.com": "sgtworkman",
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
    "agorgianitisj@hotmail.com": "johnisag",
    "phil.thomas@gametime.co": "explainanalyze",
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
    "enochlam2002@gmail.com": "eloklam",
    "eloklam@eloklam-ubuntudesktop.tail21966c.ts.net": "eloklam",
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
    # v0.16.0 additions
    "teknium@nous.dev": "teknium1",
    "alaamohanad169@gmail.com": "alaamohanad169-ship-it",
    "archer@ouyangdeMac-mini.local": "Archerouyang",  # display name 欧阳
    "batosk2@gmail.com": "Sarbai",  # git email for PR #33438 author (display: Брагарник Дмитро)
    "info@aminvakil.com": "aminvakil",
    "nikpolale@gmail.com": "polnikale",
    "sarveshagl1327@gmail.com": "sarvesh1327",  # salvaged via #38655
    "sohyuanchin@gmail.com": "wysie",
    "bedirhan@codeway.co": "bedirhancode",
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
    "zwcf5200@163.com": "zwcf5200",  # PR #38661 (SSH remote cwd fix)
    "24110240104@m.fudan.edu.cn": "YuShu",  # co-author only
    "charliekerfoot@gmail.com": "CharlieKerfoot",  # PR #18951
    # Debug share upload-time redaction (May 2026)
    "dhuysamen@gmail.com": "GodsBoy",  # PR #19318
    "github@nadyahermes.anonaddy.com": "ruangraung",  # PR #42308
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
    "275835513+paulb26@users.noreply.github.com": "paulb26",  # PR #24135 salvage (pty-bridge killpg)
    "promptsiren@gmail.com": "firefly",  # PR #18123 salvage of #16660 (ContextVars)
    "wtyopenclaw@gmail.com": "WuTianyi123",  # PR #20275 salvage of #13723 (feishu markdown)
    "zhicheng.han@mathematik.uni-goettingen.de": "hanzckernel",  # PR #20311 (api-server approval events)
    "agentsmithlaor@gmail.com": "oferlaor",  # PR #22356 salvage (cron origin sender identity)
    "jhin.lee@unity3d.com": "leehack",  # PR #22053 salvage (telegram DM topic reply fallback)
    "caojiguang@gmail.com": "caojiguang",  # PR #35117 carries #31853 (weixin _api_post/_api_get wait_for)
    "gooku94123@gmail.com": "goku94123",  # PR #46609 salvage (MiniMax reasoning extra_body)
    # pander: empty email, salvaged via PR #19665 from #16126 by @ms-alan
    "chaithanya.kumar42a@gmail.com": "chaithanyak42",  # PR #15624
    "kartik.labhshetwar@mem0.ai": "kartik-mem0",  # PR #15624
    "ayman.a.kamal@hotmail.com": "A-kamal",  # PR #18678 (xAI image resolution fix)
    # Kanban bug-fix batch salvage (May 2026)
    "frowte3k@gmail.com": "Frowtek",  # salvage of #23206 (gateway --board auto-subscribe)
    "sylw3st3rr@gmail.com": "Sylw3ster",  # salvage of #23252 (HERMES_KANBAN_BOARD restore)
    "hello@dominikh.com": "dmnkhorvath",  # salvage of #23358 (kanban worker send_message)
    "413011+smwbev@users.noreply.github.com": "smwbev",  # salvage of #23659 (aria-label colLabel)
    "58116817+TurgutKural@users.noreply.github.com": "TurgutKural",  # salvage of #23356 (HERMES_HOME inject)
    "openclaw@agent.local": "29206394",  # PR #22194 salvage (sudo -S brute-force guard, #9590)
    "freedemon@gmail.com": "fr33d3m0n",  # PR #21128 salvage (sudo stdin/askpass DANGEROUS, #17873 cat 4)
    "zhaowh3613@outlook.com": "VinceZcrikl",  # PR #23647 salvage (npm UTF-8 decode on GBK Windows)
    "abcdjmm970703@gmail.com": "JabberELF",  # PR #20238 seed (session_search dual-mode, evolved into single-shape)
    "anton.kuenzi@gmail.com": "ZeterMordio",  # PR #11754 salvage (zsh completion compdef + _arguments syntax)
    "23yntong@stu.edu.cn": "iuyup",  # PR #6155 salvage (shell=True hardening)
    "86501179+1RB@users.noreply.github.com": "1RB",  # PR #25462 salvage (discord forwarded messages)
    "44045943+ayushere@users.noreply.github.com": "ayushere",  # PR #25342 salvage (memory teardown leak)
    "15791290+domtriola@users.noreply.github.com": "domtriola",  # PR #25424 salvage (docs tirith link)
    "tuancookiez@gmail.com": "tuancookiez-hub",  # PR #34865 salvage (LSP Windows .cmd shim spawn, #34864)
    "284216128+ephron-ren@users.noreply.github.com": "ephron-ren",  # PR #25358 salvage (MiMo reasoning echo-back)
    "96843562+freqyfreqy@users.noreply.github.com": "freqyfreqy",  # PR #25423 salvage (docs LSP worktree -> repo)
    "54306477+fu576@users.noreply.github.com": "fu576",  # PR #25369 salvage (api_mode not inherited cross-provider)
    "258095375+kfa-ai@users.noreply.github.com": "kfa-ai",  # PR #25398 salvage (whatsapp quoted reply metadata)
    "99181308+magic524@users.noreply.github.com": "magic524",  # PR #25361 salvage (QQBot reconnect loop)
    "9150277+PaTTeeL@users.noreply.github.com": "PaTTeeL",  # PR #25359 salvage (custom_providers in compression length)
    "1700913+pearjelly@users.noreply.github.com": "pearjelly",  # PR #25388 salvage (feishu ws connect override sync)
    "100820567+raymaylee@users.noreply.github.com": "raymaylee",  # PR #25394 salvage (context compaction status)
    "122434621+Tianyu199509@users.noreply.github.com": "Tianyu199509",  # PR #25421 salvage (gateway PID Windows)
    "58224596+HxT9@users.noreply.github.com": "HxT9",  # PR #25760 salvage (web sync-assets cross-platform)
    "120411712+evgyur@users.noreply.github.com": "evgyur",  # PR #25651 salvage (docs media session context)
    "36507055+AsoTora@users.noreply.github.com": "AsoTora",  # PR #25624 salvage (MCP auth no-retry)
    "98992931+oxngon@users.noreply.github.com": "oxngon",  # PR #25603 salvage (forward image attachments to bg tasks)
    "37467487+yifengingit@users.noreply.github.com": "yifengingit",  # PR #25589 salvage (AUTOINCREMENT id ordering)
    "89525629+vanthinh6886@users.noreply.github.com": "vanthinh6886",  # PR #25562 salvage (.env 0600 perms)
    "16034932+Arkmusn@users.noreply.github.com": "Arkmusn",  # PR #25559 salvage (approvals.timeout from config)
    "nidhi2894@gmail.com": "nidhi-singh02",  # PR #2752 salvage (slack whitespace-only IndexError guard)
    "38173192+nidhi-singh02@users.noreply.github.com": "nidhi-singh02",
    "Jaaneek@users.noreply.github.com": "Jaaneek",  # PR #26457 (xAI Grok OAuth provider)
    # v0.14.0 additions
    "chuang.guo@hopechart.com": "wuwuzhijing",  # PR #21063 salvage (gateway docs mention Weixin)
    "nightcityblade@gmail.com": "nightcityblade",  # PR #24138 (docs voice/tts table)
    "pol.kuijken@gmail.com": "polkn",  # PR #6136 salvage (skill_view collision refusal)
    "robin@soal.org": "rewbs",
    # batch salvage (May 2026 LHF run)
    "sauravsejal40@gmail.com": "Saurav0989",  # PR #27071 (docs: hermes-eval community link)
    "220110965+Saurav0989@users.noreply.github.com": "Saurav0989",
    "aviarchi1994@gmail.com": "avifenesh",  # PR #25902 (docs: computer-use-linux MCP)
    "55848801+avifenesh@users.noreply.github.com": "avifenesh",
    "279959838+BROCCOLO1D@users.noreply.github.com": "BROCCOLO1D",  # PR #26796 (docs: spotify + HA)
    "m@matthewlai.ca": "matthewlai",  # PR #25293 (feat: gemma 4 reasoning allowlist)
    "4296245+matthewlai@users.noreply.github.com": "matthewlai",
    "109617724+0xchainer@users.noreply.github.com": "0xchainer",  # PR #27154/27138/27147 salvage
    "201800237+kronexoi@users.noreply.github.com": "kronexoi",  # PR #27167 salvage (Teams port fallback)
    "283442588+EloquentBrush0x@users.noreply.github.com": "EloquentBrush0x",  # PR #26642 salvage (post_setup parity)
    # batch salvage (May 2026 LHF run, group 2)
    "shellybotmoyer@example.com": "shellybotmoyer",  # PR #26661 (kanban --severity >=)
    "coulson@shellybotmoyer.com": "shellybotmoyer",  # PR #25576 (credential_pool ISO rehydrate)
    "258858106+shellybotmoyer@users.noreply.github.com": "shellybotmoyer",
    "33156212+ether-btc@users.noreply.github.com": "ether-btc",  # PR #26632 (memory provider whitespace guard)
    "Bloomtonjovish@gmail.com": "LifeJiggy",  # PR #26516 (paste collapse logging)
    "141562589+LifeJiggy@users.noreply.github.com": "LifeJiggy",
    "192385615+LifeJiggy@users.noreply.github.com": "LifeJiggy",  # stale salvage commit alias (PR #28315)
    "beastant1@gmail.com": "nekwo",  # PR #26481 (PS5.1 UTF-8 BOM)
    "43717185+nekwo@users.noreply.github.com": "nekwo",
    "9785479+stepanov1975@users.noreply.github.com": "stepanov1975",  # PR #22074 (setup config picker writes)
    "67979730+flooryyyy@users.noreply.github.com": "flooryyyy",  # PR #26374 (tool_trace error detection)
    "188585318+dgians@users.noreply.github.com": "dgians",  # PR #26034 (.ts/.py/.sh docs types)
    "zealy@tz.co": "dgians",  # PR #26034 (bot-committed by zealy-tzco under dgians' PR)
    "mottei.survive@gmail.com": "flanny7",  # PR #27030 (setup_open_webui python var)
    "20530505+flanny7@users.noreply.github.com": "flanny7",
    "hermesagent26@gmail.com": "hermesagent26",  # PR #26438 (kimi model-name reasoning pad)
    "276067471+hermesagent26@users.noreply.github.com": "hermesagent26",
    "71590782+kriscolab@users.noreply.github.com": "kriscolab",  # PR #26926 (deepseek default_aux_model)
    # batch salvage (May 2026 LHF run, group 3)
    "darvsum@users.noreply.github.com": "darvsum",  # PR #26766 (preserve discover_models in normalize)
    "peter@Peters-Mac-mini.local": "hueilau",  # PR #26498 (strip image parts for non-vision)
    "33933019+hueilau@users.noreply.github.com": "hueilau",
    "32297275+Timur00Kh@users.noreply.github.com": "Timur00Kh",  # PR #27114 (telegram DM topic for synthetic events)
    "al.bellemare@gmail.com": "Grogger",  # PR #27061 (windows console flash suppress)
    "7065068+Grogger@users.noreply.github.com": "Grogger",
    "18091625+Grogger@users.noreply.github.com": "Grogger",  # stale salvage commit alias (PR #28330)
    "clement@nousresearch.com": "lemassykoi",  # PR #27042 (model-switch probe keyless providers)
    "16377344+lemassykoi@users.noreply.github.com": "lemassykoi",
    "draplater@icloud.com": "draplater",  # PR #26707 (goal judge current time)
    "6349758+draplater@users.noreply.github.com": "draplater",
    "pr7426@users.noreply.github.com": "pr7426",  # PR #27048 (cron parallel job loss)
    "rahulnilvan43@gmail.com": "therahul-yo",  # PR #26215 (mock keychain in tests)
    "kingsleyemeka117@gmail.com": "flamiinngo",  # PR #27205 (UnicodeEncodeError footgun checker)
    # batch salvage (May 2026 LHF run, group 4)
    "283442588+EloquentBrush0x@users.noreply.github.com": "EloquentBrush0x",  # PR #26657 (trust_env aiohttp)
    "205509009+subtract0@users.noreply.github.com": "subtract0",  # PR #25658 (zsh $status -> $rc)
    "patryk@jarmakowicz.me": "zwolniony",  # PR #26961 (gemini x-goog-api-key)
    "12735938+zwolniony@users.noreply.github.com": "zwolniony",
    "ambuj@dodopayments.com": "that-ambuj",  # PR #26582 (preserve underscores)
    "zccyman@163.com": "zccyman",  # PR #25294 (custom provider api_key_env alias)
    # xAI cluster batch salvage (May 2026)
    "lgndscntn@gmail.com": "Fewmanism",  # PR #27420 (threaded xAI OAuth callback)
    "slimydog@Faisals-Mac-mini.local": "Slimydog21",  # PR #28021 (strip slash enums xAI Responses)
    "194121339+Slimydog21@users.noreply.github.com": "Slimydog21",  # PR #28021 salvage (noreply form)
    "bitkyc08@gmail.com": "lidge-jun",  # PR #26814 (api server browser security headers)
    "sp_ps@Mac-mini.lan": "phoenixshen",  # PR #26768 (respect user-configured vision model)
    "1594534+phoenixshen@users.noreply.github.com": "phoenixshen",
    "147827411+AhmetArif0@users.noreply.github.com": "AhmetArif0",  # PR #26635 (line proxy env vars)
    # batch salvage (May 2026 LHF run, group 5)
    "hari@Hariharans-MacBook-Air-8.local": "haran2001",  # PR #27070 (i18n catalog test)
    "hariharan15151@gmail.com": "haran2001",  # PR #27068 (qwen3.6-plus 1M context)
    "56040092+haran2001@users.noreply.github.com": "haran2001",
    "1472110+ms-alan@users.noreply.github.com": "ms-alan",  # PR #26443 (reload-skills tab completion)
    "ganlinbupt@gmail.com": "godlin-gh",  # PR #26118 (ACP polished tools)
    "wesley.simplicio.ext@siemens-energy.com": "wesleysimplicio",  # PR #25777 (xterm.js native selection)
    "6108320+wesleysimplicio@users.noreply.github.com": "wesleysimplicio",
    "carryzuo00@gmail.com": "Carry00",  # PR #26851 (doctor SSH env vars)
    "alaamohanad169-ship-it@users.noreply.github.com": "alaamohanad169-ship-it",  # PR #26036 (telegram typing after send)
    "vigo@hermes": "hawknewton",  # PR #26294 (bedrock boto3 lazy_deps)
    "211668+hawknewton@users.noreply.github.com": "hawknewton",
    "quenvix00@gmail.com": "QuenVix",  # PR #26761/26772 salvage
    "164776164+QuenVix@users.noreply.github.com": "QuenVix",
    "262945885+Mind-Dragon@users.noreply.github.com": "Mind-Dragon",  # PR #26966 salvage
    "soynchuux@gmail.com": "soynchux",  # PR #27060 salvage
    "209694554+soynchux@users.noreply.github.com": "soynchux",
    # batch salvage (May 2026 LHF run, group 6 — final)
    "6666242+bird@users.noreply.github.com": "bird",  # PR #25219 (gateway docker exit-75 restart)
    "david@loadmagic.ai": "davidcampbelldc",  # PR #26834 (web_server proxy_headers=False)
    "165905879+davidcampbelldc@users.noreply.github.com": "davidcampbelldc",
    "hoangv.pham0803@gmail.com": "hehehe0803",  # PR #26212 salvage (codex kanban writable root)
    "26063003+hehehe0803@users.noreply.github.com": "hehehe0803",
    "kasunvinod@users.noreply.github.com": "kasunvinod",  # PR #24126 salvage (codex timeout propagation)
    "15059870+kasunvinod@users.noreply.github.com": "kasunvinod",
    "38348871+vaddisrinivas@users.noreply.github.com": "vaddisrinivas",  # PR #26394 salvage (Docker messaging extra)
    # batch salvage (May 2026 LHF run, group 7)
    "198679067+02356abc@users.noreply.github.com": "02356abc",  # PR #28286 salvage (wecom CLOSING)
    "1743117+burjorjee@users.noreply.github.com": "burjorjee",  # PR #28201 salvage (inline-shell timeout guard)
    "keki@MacBookPro.attlocal.net": "burjorjee",
    "264690993+oseftg@users.noreply.github.com": "oseftg",  # PR #28168 salvage (natural ending emoji/caret)
    "hex.hermes@agentmail.to": "oseftg",
    "236912655+rudi193-cmd@users.noreply.github.com": "rudi193-cmd",  # PR #28241 salvage (empty credential pool)
    "rudi193@gmail.com": "rudi193-cmd",
    "86684667+sadiksaifi@users.noreply.github.com": "sadiksaifi",  # PR #27982 salvage (kanban horiz scroll)
    "mail@sadiksaifi.dev": "sadiksaifi",
    "231588442+vynxevainglory-ai@users.noreply.github.com": "vynxevainglory-ai",  # PR #29233 salvage (kanban scrollbar + body overflow)
    "vynxevainglory@gmail.com": "vynxevainglory-ai",
    # batch salvage (May 2026 LHF run, group 8)
    "266824395+AceWattGit@users.noreply.github.com": "AceWattGit",  # PR #28159 salvage (_pool_may_recover NameError)
    "57024493+YuanHanzhong@users.noreply.github.com": "YuanHanzhong",  # PR #28032 salvage (x.com status link-like)
    "24368158+colin-chang@users.noreply.github.com": "colin-chang",  # PR #28245/#28249/#28251 salvage
    "zhangcheng5468@gmail.com": "colin-chang",
    "172729123+felix-windsor@users.noreply.github.com": "felix-windsor",  # PR #28019 salvage (cron asterisks)
    "felixwindsor3344@gmail.com": "felix-windsor",
    "259054917+houenyang-momo@users.noreply.github.com": "houenyang-momo",  # PR #28205 salvage (charizard contrast)
    "33547839+sir-ad@users.noreply.github.com": "sir-ad",  # PR #31941 salvage (compaction noise)
    "adarsh.agrahari26@gmail.com": "sir-ad",
    "269599864+rdasilva1016-ui@users.noreply.github.com": "rdasilva1016-ui",  # PR #31098 salvage (Telegram /start ping)
    "rdasilva1016-ui@users.noreply.github.com": "rdasilva1016-ui",
    "35931201+iqdoctor@users.noreply.github.com": "iqdoctor",  # PR #28095 salvage (windows installer docs)
    "29513231+joe102084@users.noreply.github.com": "joe102084",  # PR #28151 salvage (whitespace cron responses)
    "joe102084@gmail.com": "joe102084",
    "4139778+jvinals@users.noreply.github.com": "jvinals",  # PR #27936 salvage (Slack U-IDs)
    "3001335+maxmilian@users.noreply.github.com": "maxmilian",  # PR #28267 salvage (Change Model portal)
    "maxmilian@gmail.com": "maxmilian",
    "41468846+samggggflynn@users.noreply.github.com": "samggggflynn",  # PR #27952 salvage (dingtalk pre_start)
    "abc401011721@gmail.com": "samggggflynn",
    "yannsunn@users.noreply.github.com": "yannsunn",  # PR #28064 salvage (xai proxy upstream)
    "yannsunn1116@gmail.com": "yannsunn",
    "asdlem@users.noreply.github.com": "asdlem",  # PR #27852 salvage (clarify full text in body)
    # batch salvage (May 2026 LHF run, group 9)
    "1779909+jdelmerico@users.noreply.github.com": "jdelmerico",  # PR #28278 salvage (signal require_mention)
    "20639347+justemu@users.noreply.github.com": "justemu",  # PR #27996 salvage (matrix thread_require_mention)
    "justemu@users.noreply.github.com": "justemu",
    "57024493+YuanHanzhong@users.noreply.github.com": "YuanHanzhong",  # PR #28029 salvage (dashboard scrollback)
    "YuanHanzhong@users.noreply.github.com": "YuanHanzhong",
    "1663402+noctilust@users.noreply.github.com": "noctilust",  # PR #28080 salvage (stale TUI resume env)
    "1663402+freeurmind@users.noreply.github.com": "noctilust",
    "35164907+MoonJuhan@users.noreply.github.com": "MoonJuhan",  # PR #28288 salvage (unreadable JSONL transcripts)
    "codemike@naver.com": "MoonJuhan",
    "201563152+outsourc-e@users.noreply.github.com": "outsourc-e",  # PR #28164 salvage (cron emoji ZWJ)
    "201803425+Zyrixtrex@users.noreply.github.com": "Zyrixtrex",  # PR #28275 salvage (Google OAuth timeout)
    "zyrixtrex@gmail.com": "Zyrixtrex",
    "120500656+ooovenenoso@users.noreply.github.com": "ooovenenoso",  # PR #28256 salvage (tool loop recovery hints)
    "120500656+oooindefatigable@users.noreply.github.com": "ooovenenoso",
    "vanthinh6886@gmail.com": "vanthinh6886",  # PR #28018 salvage (yaml/flock/atomic write guards)
    "erik.engervall@gmail.com": "erikengervall",  # PR #28774 (firecrawl integration tag)
    "egilewski@egilewski.com": "egilewski",  # PR #30432 (MEDIA path traversal fix, GHSA-jmf9-9729-7pp8)
    "edison@mcclean.codes": "McClean-Edison",  # PR #29817 (register_auxiliary_task plugin API)
    "zhangsamuel12@gmail.com": "SamuelZ12",  # PR #7480 (show recap after in-session resume)
    "490408354@qq.com": "daizhonggeng",  # PR #9020 (numbered /resume selection)
    "claw@openclaw.ai": "wanwan2qq",  # PR #10215 (strip brackets/quotes from /resume; gateway session-ID lookup)
    "simo.kiihamaki@gmail.com": "SimoKiihamaki",  # PR #30773 (Windows /reset+/new freeze; stdin fallback for modal)
    "66773372+Tranquil-Flow@users.noreply.github.com": "Tranquil-Flow",  # PR #27518 (bracketed-paste timeout)
    "8bit64k@pm.me": "8bit64k",  # PR #14681 (TUI /q alias from quit to queue)
    "chenglunhu@gmail.com": "hclsys",  # PR #31985 (TUI /q alias regression test)
    "dearmayo@localhost": "ffr31mr",  # PR #32103 (SubdirectoryHintTracker workspace boundary)
    "TheOnlyMika@users.noreply.github.com": "TheOnlyMika",  # PR #32155 (dashboard XSS + defusedxml)
    "krislidimo@gmail.com": "krislidimo",  # PR #29775 (tighten Telegram table row-group spacing; drop redundant first bullet)
    "timothy.b.dixon@gmail.com": "Codename-11",  # PR #29302 (API server session controls — sessions/chat/fork/stream)
    "jpschwartz2@uwalumni.com": "Schwartz10",  # PR #29302 sub-PR (multimodal media in session chat API)
    "JohnC1009@users.noreply.github.com": "JohnC1009",  # PR #32020 salvage (auth: global auth.json fallback in _load_provider_state)
    "biser@bisko.be": "bisko",  # PR #33784 salvage (re-pad reasoning_content on cross-provider fallback to require-side providers)
    # v0.15.0 additions
    "glen@workmanfirearms.com": "sgtworkman",
    "jorge.fuenmayort@gmail.com": "jfuenmayor",
    "josh.dow@prepad.io": "joshuadow",  # PR #43004 salvage (desktop WS session rebind)
    "mordred@inaugust.com": "emonty",
    "rodrigoeq@hotmail.com": "rodrigoeqnit",
    "soliva.johnpaul@icloud.com": "jonpol01",
    "2182712990@qq.com": "yu-xin-c",  # PR #32122 (Docker audio bridge notes)
    "baxter@bitreserve.ai": "BaxBit",  # PR #30200 (Svix webhook signature validation)
    "chris.eth@qq.com": "duyua9",  # PR #10949 (render object config values structurally)
    "ethie@nous": "ethernet8023",  # PR #29342 (TUI clipboard copy on linux/wayland)
    "jiahuigu@sjtu.edu.cn": "Jiahui-Gu",  # PR #29276 (guard pickle.loads in darwinian-evolver)
    "justinccdev@gmail.com": "justincc",  # PR #28914 (set tool_name on tool-result messages)
    "kdkcfp@gmail.com": "slowtokki0409",  # PR #29025 (ignore local Hermes runtime files)
    "peter.yuqin@gmail.com": "WuKongAI-CMU",  # PR #10082 (reject symlinked audio inputs)
    "sunil.nitie@gmail.com": "Sunil123135",  # PR #31031 (Windows Docker Desktop compose)
    "weichangyuwcy@gmail.com": "ChyuWei",  # PR #30987 (TUI TTS env var on voice off)
    # batch salvage PR #35758 (perf micro-fixes)
    "116212274+amathxbt@users.noreply.github.com": "amathxbt",  # PR #22155 (cache tool_output_limits)
    "takis312@hotmail.com": "ErnestHysa",  # PRs #32636/#32708 (MCP asyncio.sleep + O(n^2) watcher drain)
    "me@simontaggart.com": "SiTaggart",  # PR #35583 (docker_forward_env empty-secret .env fallback)
    "2663402852@qq.com": "x1am1",  # PR #35098 (chown root-owned top-level HERMES_HOME state files)
    "nicsequenzy@gmail.com": "polnikale",  # PR #35717 (discover Playwright headless_shell browser)
    "wasdhkzk@gmail.com": "whyhkzk",  # PR #32407 (sandbox-mirror inner-container guard; commits authored as whyhkzk + zhukun)
    "leonard@sellem.me": "leonardsellem",  # PR #37405 (desktop WS origin guard on remote/Tailscale binds)
    "42903577+ohMyJason@users.noreply.github.com": "ohMyJason",  # PR #29810 (discover_models in custom_providers section 4)
    "singhsanidhya741@gmail.com": "sanidhyasin",  # PR #40403 salvage (model.default_headers for custom OpenAI-compatible providers, #40033)
    "josephjohnson.joel@gmail.com": "JoelJJohnson",  # PR #39913 salvage (Windows ConPTY dashboard chat bridge)
    "andreas@schwarz-ketsch.de": "Nea74",  # PR #40022 co-author credit (same Windows ConPTY bridge design)
    "chanhokyim@gmail.com": "joel611",  # PR #33958 salvage (DISCORD_ALLOWED_ROLES role_authorized gateway flag)
    "desg38@gmail.com": "dschnurbusch",  # PR #42373 salvage (archive compressed conversation lineages)
    "bsmith@bramarstrategicservices.com": "bcsmith528",  # PR #20589 salvage (register_slack_action_handler plugin API)
    "sunsky.lau@gmail.com": "liuhao1024",  # PR #45494 salvage (claim session slot before auto-resume task; #45456)
    "andrewdmwalker@gmail.com": "capt-marbles",  # PR #38440 salvage (resolve xAI OAuth credentials across profiles; #43589)
    "infinitycrew39@gmail.com": "infinitycrew39",  # PR #47945 salvage (scope langfuse trace state by turn/request ids; #48292)
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

    # Keep the desktop Electron app's package.json version in lockstep with the
    # Python package version. The desktop About panel reads the live Hermes
    # version at runtime, but app.getVersion()/packaging metadata still come
    # from this field, so it must track pyproject to avoid drift.
    desktop_pkg = REPO_ROOT / "apps" / "desktop" / "package.json"
    if desktop_pkg.exists():
        pkg_text = desktop_pkg.read_text(encoding="utf-8")
        pkg_text = re.sub(
            r'("version"\s*:\s*)"[^"]+"',
            rf'\g<1>"{semver}"',
            pkg_text,
            count=1,
        )
        desktop_pkg.write_text(pkg_text, encoding="utf-8")

    # Update ACP Registry manifest + npm launcher (must stay version-locked
    # with pyproject — enforced by tests/acp/test_registry_manifest.py).
    _update_acp_registry_versions(semver)


def _update_acp_registry_versions(semver: str) -> None:
    """Bump the ACP Registry manifest's version + uvx package pin in lockstep
    with pyproject.

    Skips silently if the manifest is missing — older release branches predate
    the ACP Registry assets.
    """
    if ACP_REGISTRY_MANIFEST.exists():
        manifest = json.loads(ACP_REGISTRY_MANIFEST.read_text(encoding="utf-8"))
        manifest["version"] = semver
        uvx = manifest.get("distribution", {}).get("uvx", {})
        if "package" in uvx:
            uvx["package"] = f"hermes-agent[acp]=={semver}"
        # Preserve trailing newline + 2-space indent the file already uses.
        ACP_REGISTRY_MANIFEST.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def build_release_artifacts(semver: str) -> list[Path]:
    """Build sdist/wheel artifacts for the current release.

    Tries ``uv build`` first (matching the CI workflow), falls back to
    ``python -m build`` if uv is unavailable.
    """
    dist_dir = REPO_ROOT / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)

    # Prefer uv build (matches CI workflow), fall back to python -m build.
    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "build", "--sdist", "--wheel"]
    else:
        cmd = [sys.executable, "-m", "build", "--sdist", "--wheel"]

    result = subprocess.run(
        cmd,
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
        print("    Install uv or the 'build' package to attach sdist/wheel assets.")
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

    # Format: hash<US>author_name<US>author_email<US>subject\0body
    # Using %x1f (unit separator) to avoid conflict with | in author names
    log = git(
        "log", range_spec,
        "--format=%H%x1f%an%x1f%ae%x1f%s%x00%b%x00",
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
        # Split on first null to separate "hash<US>name<US>email<US>subject" from "body"
        if "\0" in entry:
            header, body = entry.split("\0", 1)
            body = body.strip()
        else:
            header = entry
            body = ""
        parts = header.split("\x1f", 3)
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


def get_pr_number(subject: str) -> str | None:
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
        return

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
    print(f"  Unique authors:  {len({c['github_author'] for c in commits})}")
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
            add_files = [str(VERSION_FILE), str(PYPROJECT_FILE)]
            if ACP_REGISTRY_MANIFEST.exists():
                add_files.append(str(ACP_REGISTRY_MANIFEST))
            add_result = git_result("add", *add_files)
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
        changelog_file.write_text(changelog, encoding="utf-8")

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
