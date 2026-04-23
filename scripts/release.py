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
    "teknium@nousresearch.com": "teknium1",
    "127238744+teknium1@users.noreply.github.com": "teknium1",
    # contributors (from noreply pattern)
    "wangqiang@wangqiangdeMac-mini.local": "xiaoqiang243",
    "snreynolds2506@gmail.com": "snreynolds",
    "35742124+0xbyt4@users.noreply.github.com": "0xbyt4",
    "71184274+MassiveMassimo@users.noreply.github.com": "MassiveMassimo",
    "massivemassimo@users.noreply.github.com": "MassiveMassimo",
    "82637225+kshitijk4poor@users.noreply.github.com": "kshitijk4poor",
    "keifergu@tencent.com": "keifergu",
    "kshitijk4poor@users.noreply.github.com": "kshitijk4poor",
    "abner.the.foreman@agentmail.to": "Abnertheforeman",
    "harryykyle1@gmail.com": "hharry11",
    "kshitijk4poor@gmail.com": "kshitijk4poor",
    "16443023+stablegenius49@users.noreply.github.com": "stablegenius49",
    "185121704+stablegenius49@users.noreply.github.com": "stablegenius49",
    "101283333+batuhankocyigit@users.noreply.github.com": "batuhankocyigit",
    "255305877+ismell0992-afk@users.noreply.github.com": "ismell0992-afk",
    "valdi.jorge@gmail.com": "jvcl",
    "francip@gmail.com": "francip",
    "omni@comelse.com": "omnissiah-comelse",
    "oussama.redcode@gmail.com": "mavrickdeveloper",
    "126368201+vilkasdev@users.noreply.github.com": "vilkasdev",
    "137614867+cutepawss@users.noreply.github.com": "cutepawss",
    "96793918+memosr@users.noreply.github.com": "memosr",
    "milkoor@users.noreply.github.com": "milkoor",
    "xuerui911@gmail.com": "Fatty911",
    "131039422+SHL0MS@users.noreply.github.com": "SHL0MS",
    "77628552+raulvidis@users.noreply.github.com": "raulvidis",
    "145567217+Aum08Desai@users.noreply.github.com": "Aum08Desai",
    "256820943+kshitij-eliza@users.noreply.github.com": "kshitij-eliza",
    "44278268+shitcoinsherpa@users.noreply.github.com": "shitcoinsherpa",
    "104278804+Sertug17@users.noreply.github.com": "Sertug17",
    "112503481+caentzminger@users.noreply.github.com": "caentzminger",
    "258577966+voidborne-d@users.noreply.github.com": "voidborne-d",
    "sir_even@icloud.com": "sirEven",
    "36056348+sirEven@users.noreply.github.com": "sirEven",
    "70424851+insecurejezza@users.noreply.github.com": "insecurejezza",
    "254021826+dodo-reach@users.noreply.github.com": "dodo-reach",
    "259807879+Bartok9@users.noreply.github.com": "Bartok9",
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
    "nocoo@users.noreply.github.com": "nocoo",
    "30841158+n-WN@users.noreply.github.com": "n-WN",
    "tsuijinglei@gmail.com": "hiddenpuppy",
    "jerome@clawwork.ai": "HiddenPuppy",
    "leoyuan0099@gmail.com": "keyuyuan",
    "bxzt2006@163.com": "Only-Code-A",
    "i@troy-y.org": "TroyMitchell911",
    "mygamez@163.com": "zhongyueming1121",
    "hansnow@users.noreply.github.com": "hansnow",
    "134848055+UNLINEARITY@users.noreply.github.com": "UNLINEARITY",
    "ben.burtenshaw@gmail.com": "burtenshaw",
    "roopaknijhara@gmail.com": "rnijhara",
    # contributors (manual mapping from git names)
    "ahmedsherif95@gmail.com": "asheriif",
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
    "dean.kerr@gmail.com": "deankerr",
    "socrates1024@gmail.com": "socrates1024",
    "seanalt555@gmail.com": "Salt-555",
    "satelerd@gmail.com": "satelerd",
    "numman.ali@gmail.com": "nummanali",
    "0xNyk@users.noreply.github.com": "0xNyk",
    "0xnykcd@googlemail.com": "0xNyk",
    "buraysandro9@gmail.com": "buray",
    "contact@jomar.fr": "joshmartinelle",
    "camilo@tekelala.com": "tekelala",
    "vincentcharlebois@gmail.com": "vincentcharlebois",
    "aryan@synvoid.com": "aryansingh",
    "johnsonblake1@gmail.com": "blakejohnson",
    "hcn518@gmail.com": "pedh",
    "haileymarshall005@gmail.com": "haileymarshall",
    "greer.guthrie@gmail.com": "g-guthrie",
    "kennyx102@gmail.com": "bobashopcashier",
    "shokatalishaikh95@gmail.com": "areu01or00",
    "bryan@intertwinesys.com": "bryanyoung",
    "christo.mitov@gmail.com": "christomitov",
    "hermes@nousresearch.com": "NousResearch",
    "hermes@noushq.ai": "benbarclay",
    "chinmingcock@gmail.com": "ChimingLiu",
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
    "oluwadareab12@gmail.com": "bennytimz",
    "simon@simonmarcus.org": "simon-marcus",
    "xowiekk@gmail.com": "Xowiek",
    "1243352777@qq.com": "zons-zhaozhy",
    "e.silacandmr@gmail.com": "Es1la",
    # ── bulk addition: 75 emails resolved via API, PR salvage bodies, noreply
    #    crossref, and GH contributor list matching (April 2026 audit) ──
    "1115117931@qq.com": "aaronagent",
    "1506751656@qq.com": "hqhq1025",
    "364939526@qq.com": "luyao618",
    "hgk324@gmail.com": "houziershi",
    "176644217+PStarH@users.noreply.github.com": "PStarH",
    "51058514+Sanjays2402@users.noreply.github.com": "Sanjays2402",
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
    "vominh1919@users.noreply.github.com": "vominh1919",
    "iamagenius00@users.noreply.github.com": "iamagenius00",
    "9219265+cresslank@users.noreply.github.com": "cresslank",
    "trevmanthony@gmail.com": "trevthefoolish",
    "ziliangpeng@users.noreply.github.com": "ziliangpeng",
    "centripetal-star@users.noreply.github.com": "centripetal-star",
    "LeonSGP43@users.noreply.github.com": "LeonSGP43",
    "154585401+LeonSGP43@users.noreply.github.com": "LeonSGP43",
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
    "tranquil_flow@protonmail.com": "Tranquil-Flow",
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
    "mbelleau@Michels-MacBook-Pro.local": "malaiwah",
    "michel.belleau@malaiwah.com": "malaiwah",
    "gnanasekaran.sekareee@gmail.com": "gnanam1990",
    "jz.pentest@gmail.com": "0xyg3n",
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
    "anthhub@163.com": "anthhub",
    "shenuu@gmail.com": "shenuu",
    "xiayh17@gmail.com": "xiayh0107",
    "zhujianxyz@gmail.com": "opriz",
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
        Path(args.output).write_text(changelog)
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
