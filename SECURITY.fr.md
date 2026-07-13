# Politique de sécurité de Hermes Agent

Ce document décrit le modèle de confiance de Hermes Agent, identifie
l'unique frontière de sécurité que le projet considère comme structurelle
et définit le périmètre des signalements de vulnérabilités.

## 1. Signaler une vulnérabilité

Signalez de manière privée via les [GitHub Security Advisories](https://github.com/NousResearch/hermes-agent/security/advisories/new)
ou **security@nousresearch.com**. N'ouvrez pas d'issues publiques pour
des vulnérabilités de sécurité. **Hermes Agent n'opère pas de programme
de bug bounty.**

Un signalement utile comprend :

- Une description concise et une évaluation de la gravité.
- Le composant affecté, identifié par chemin de fichier et plage de
  lignes (ex. `path/to/file.py:120-145`).
- Les détails de l'environnement (`hermes version`, SHA du commit, OS,
  version de Python).
- Une reproduction sur `main` ou la dernière release.
- Une indication de la frontière de confiance du §2 qui est franchie.

Merci de lire le §2 et le §3 avant d'envoyer. Les signalements qui
démontrent les limites d'une heuristique intra-processus que cette
politique ne considère pas comme une frontière seront fermés comme hors
périmètre au titre du §3 — mais voyez le §3.2 : ils restent les
bienvenus sous forme d'issues ou de pull requests classiques, simplement
pas via le canal de sécurité privé.

---

## 2. Modèle de confiance

Hermes Agent est un agent personnel mono-utilisateur. Sa posture est en
couches, et ces couches n'ont pas toutes le même poids. Les rapporteurs
et les opérateurs doivent raisonner à leur sujet dans les mêmes termes.

### 2.1 Définitions

- **Processus de l'agent.** L'interpréteur Python qui exécute Hermes
  Agent, y compris tout module Python qu'il a chargé (skills, plugins,
  gestionnaires de hooks).
- **Backend de terminal.** Une cible d'exécution enfichable pour
  l'outil `terminal()`. Le backend par défaut exécute les commandes
  directement sur l'hôte. D'autres backends les exécutent dans un
  conteneur, une sandbox cloud ou un hôte distant.
- **Surface d'entrée.** Tout canal par lequel du contenu entre dans le
  contexte de l'agent : saisie de l'opérateur, requêtes web, e-mails,
  messages du gateway, lectures de fichiers, réponses de serveurs MCP,
  résultats d'outils.
- **Enveloppe de confiance.** L'ensemble des ressources auxquelles un
  opérateur a implicitement donné accès à Hermes Agent en le lançant —
  typiquement, tout ce que le compte utilisateur de l'opérateur peut
  atteindre sur l'hôte.
- **Position.** Une déclaration explicite dans la documentation ou le
  code de Hermes Agent sur la façon dont une couche consommatrice
  (adaptateur, UI, écriture de fichiers, shell) doit traiter la sortie
  de l'agent — ex. « le dashboard rend la sortie de l'agent comme du
  HTML inerte ».

### 2.2 La frontière : l'isolation au niveau de l'OS

**La seule frontière de sécurité face à un LLM adverse est le système
d'exploitation.** Rien à l'intérieur du processus de l'agent ne
constitue un confinement — ni le portail d'approbation, ni la rédaction
des sorties, ni aucun scanner de motifs, ni aucune liste d'outils
autorisés. Tout composant intra-processus qui filtre la sortie du LLM
est une heuristique opérant sur une chaîne influencée par l'attaquant,
et cette politique le traite comme tel.

Hermes Agent prend en charge deux postures d'isolation au niveau de
l'OS. Elles répondent à des menaces différentes et un opérateur doit
choisir en connaissance de cause.

#### Isolation par le backend de terminal

Un backend de terminal autre que celui par défaut exécute les commandes
shell émises par le LLM dans un conteneur, un hôte distant ou une
sandbox cloud. Les outils de fichiers (`read_file`, `write_file`,
`patch`) passent aussi par ce backend, puisqu'ils sont implémentés
au-dessus du contrat du shell — ils ne peuvent pas atteindre de chemins
que le backend n'expose pas.

Ce que cela confine : tout ce que l'agent fait en émettant des
opérations shell ou fichiers. Ce que cela ne confine **pas** : tout ce
que l'agent fait dans son propre processus Python. Cela inclut l'outil
d'exécution de code (lancé comme sous-processus sur l'hôte), les
sous-processus MCP (lancés depuis l'environnement de l'agent), le
chargement des plugins, le déclenchement des hooks et le chargement des
skills (tous importés dans l'interpréteur de l'agent).

L'isolation par le backend de terminal est la bonne posture lorsque le
risque redouté est l'émission par le LLM de commandes shell
destructrices ou d'écritures indésirables via les outils de fichiers,
et que l'opérateur est par ailleurs de confiance.

#### Encapsulation du processus complet

L'encapsulation du processus complet exécute l'intégralité de l'arbre
de processus de l'agent dans une sandbox. Chaque chemin de code —
shell, exécution de code, MCP, outils de fichiers, plugins, hooks,
chargement des skills — est soumis à la même politique de système de
fichiers, de réseau, de processus et (le cas échéant) d'inférence.

Hermes Agent le permet de deux façons :

- **L'image Docker et la configuration Compose de Hermes Agent.** Plus
  légère ; l'agent tourne dans un conteneur standard avec des montages
  et une politique réseau configurés par l'opérateur.
- **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)**.
  OpenShell fournit des sandboxes par session avec une politique
  déclarative couvrant le système de fichiers, le réseau (egress L7),
  les processus/syscalls et le routage d'inférence. Les politiques
  réseau et d'inférence sont rechargeables à chaud. Les identifiants
  sont injectés depuis un magasin Provider et ne touchent jamais le
  système de fichiers de la sandbox.

Sous une encapsulation du processus complet, les heuristiques
intra-processus de Hermes Agent (§2.4) jouent le rôle de prévention
des accidents, par-dessus une véritable frontière. C'est la posture
prise en charge lorsque l'agent ingère du contenu provenant de surfaces
que l'opérateur ne contrôle pas — le web ouvert, les e-mails entrants,
les canaux multi-utilisateurs, les serveurs MCP non fiables — ainsi que
pour les déploiements en production ou partagés.

Les opérateurs qui exécutent le backend local par défaut avec des
surfaces d'entrée non fiables, ou qui exécutent une sandbox de backend
de terminal en s'attendant à ce qu'elle confine des chemins de code qui
ne passent pas par le shell, opèrent en dehors de la posture de
sécurité prise en charge.

### 2.3 Portée des identifiants

Hermes Agent filtre l'environnement qu'il transmet à ses composants
intra-processus de moindre confiance : sous-processus shell,
sous-processus MCP, scripts de tâches cron et processus enfant
d'exécution de code. Les identifiants comme les clés API des providers
et les tokens du gateway sont retirés par défaut ; les variables
explicitement déclarées par l'opérateur ou par une skill chargée sont
transmises.

Cela réduit l'exfiltration opportuniste. Ce n'est pas du confinement.
Tout composant s'exécutant dans le processus de l'agent (skills,
plugins, gestionnaires de hooks) peut lire tout ce que l'agent lui-même
peut lire, y compris les identifiants en mémoire. La parade contre un
composant intra-processus compromis est la revue par l'opérateur avant
installation (§2.4, §2.5), pas le nettoyage de l'environnement.

### 2.4 Heuristiques intra-processus

Les composants suivants filtrent le comportement du LLM ou alertent à
son sujet. Ils sont utiles. Ce ne sont pas des frontières.

- Le **portail d'approbation** détecte les motifs shell destructeurs
  courants et demande confirmation à l'opérateur avant exécution. Le
  shell est Turing-complet ; une liste noire sur des chaînes shell est
  structurellement incomplète. Le portail attrape les erreurs en mode
  coopératif, pas les sorties adverses.
- La **rédaction des sorties** supprime de l'affichage les motifs
  ressemblant à des secrets. Un producteur de sorties déterminé la
  contournera.
- **Skills Guard** analyse le contenu des skills installables à la
  recherche de motifs d'injection. C'est une aide à la revue ; la
  frontière pour les skills tierces est la revue par l'opérateur avant
  installation. Passer une skill en revue signifie lire son code Python
  et ses scripts, pas seulement sa description SKILL.md — les skills
  exécutent du Python arbitraire au moment de l'import.

### 2.5 Modèle de confiance des plugins

Les plugins se chargent dans le processus de l'agent et s'exécutent
avec tous ses privilèges : ils peuvent lire les mêmes identifiants,
appeler les mêmes outils, enregistrer les mêmes hooks et importer les
mêmes modules que n'importe quel code livré dans le dépôt. La frontière
pour les plugins tiers est la revue par l'opérateur avant installation
— la même règle que pour les skills (§2.4), mentionnée à part parce que
les plugins sont architecturalement plus lourds et embarquent souvent
leurs propres services d'arrière-plan, écouteurs réseau et dépendances.

Un plugin malveillant ou bogué n'est pas une vulnérabilité de Hermes
Agent en soi. Les bugs dans le chemin d'installation ou de découverte
des plugins de Hermes Agent qui empêchent l'opérateur de voir ce qu'il
installe sont dans le périmètre au titre du §3.1.

### 2.6 Surfaces externes

Une **surface externe** est tout canal extérieur au processus local de
l'agent par lequel un appelant peut déclencher du travail de l'agent,
résoudre des approbations ou recevoir la sortie de l'agent. Chaque
surface a son propre modèle d'autorisation, mais les règles ci-dessous
s'appliquent uniformément.

**Surfaces dans Hermes Agent :**

- **Adaptateurs de plateforme du gateway.** Les intégrations de
  messagerie dans `gateway/platforms/` (Telegram, Discord, Slack,
  e-mail, SMS, etc.) et les adaptateurs analogues livrés sous forme de
  plugins.
- **Surfaces HTTP exposées au réseau.** L'adaptateur du serveur API, le
  plugin du dashboard, les endpoints HTTP du plugin kanban, et tout
  autre plugin qui ouvre un socket d'écoute.
- **Adaptateurs éditeur / IDE.** L'adaptateur ACP (`acp_adapter/`) et
  les intégrations équivalentes qui acceptent des requêtes d'un
  processus client local.
- **Le gateway TUI (`tui_gateway/`).** Backend JSON-RPC de l'interface
  terminal Ink, joint via IPC local.

**Règles uniformes :**

1. **Une autorisation est requise à chaque surface qui franchit une
   frontière de confiance.** Pour les surfaces de messagerie et HTTP
   réseau, la frontière est le réseau : l'autorisation prend la forme
   d'une liste d'appelants autorisés configurée par l'opérateur. Pour
   les surfaces éditeur et IPC local (ACP, gateway TUI), la frontière
   est le compte utilisateur de l'hôte : l'autorisation consiste à
   s'appuyer sur le contrôle d'accès de l'OS (permissions de fichiers,
   liaisons loopback uniquement) et à ne pas exposer la surface au-delà
   de l'utilisateur local sans une couche d'authentification réseau
   explicite.
2. **Une liste d'autorisation est requise pour chaque adaptateur exposé
   au réseau qui est activé.** Les adaptateurs doivent refuser de
   déclencher du travail de l'agent, de résoudre des approbations ou de
   relayer des sorties tant qu'aucune liste d'autorisation n'est
   définie. Les chemins de code qui laissent passer par défaut
   lorsqu'aucune liste n'est configurée sont des bugs de code dans le
   périmètre au titre du §3.1.
3. **Les identifiants de session sont des poignées de routage, pas des
   frontières d'autorisation.** Connaître l'ID de session d'un autre
   appelant ne donne pas accès à ses approbations ni à ses sorties ;
   l'autorisation est toujours revérifiée contre la liste
   d'autorisation (ou son équivalent au niveau de l'OS).
4. **Au sein de l'ensemble autorisé, tous les appelants bénéficient de
   la même confiance.** Hermes Agent ne modélise pas de capacités par
   appelant au sein d'un même adaptateur. Les opérateurs qui ont besoin
   d'une séparation des capacités doivent exécuter des instances
   d'agent distinctes avec des listes d'autorisation distinctes.
5. **Lier une surface locale à une interface autre que loopback est une
   décision d'opérateur de dernier recours (§3.2).** Le dashboard et
   les autres serveurs HTTP de plugins écoutent en loopback par
   défaut ; les exposer via `--host 0.0.0.0` ou équivalent fait du
   durcissement pour exposition publique (§4) la responsabilité de
   l'opérateur.

---

## 3. Périmètre

### 3.1 Dans le périmètre

- L'échappement d'une posture d'isolation OS déclarée (§2.2) : un
  chemin de code contrôlé par l'attaquant atteignant un état que la
  posture prétendait confiner.
- L'accès non autorisé à une surface externe : un appelant hors de
  l'ensemble d'autorisation configuré (liste d'autorisation, ou
  équivalent au niveau de l'OS pour les surfaces IPC locales) qui
  déclenche du travail, reçoit des sorties ou résout des approbations
  (§2.6).
- L'exfiltration d'identifiants : fuite d'identifiants de l'opérateur
  ou de matériel d'autorisation de session vers une destination hors de
  l'enveloppe de confiance, via un mécanisme qui aurait dû l'empêcher
  (bug de nettoyage de l'environnement, journalisation d'un adaptateur,
  erreur de transport qui déverse des identifiants vers un amont,
  etc.).
- Les violations de la documentation du modèle de confiance : du code
  qui se comporte contrairement à ce que cette politique, la propre
  documentation de Hermes Agent ou les attentes raisonnables d'un
  opérateur laisseraient prévoir — y compris les cas où Hermes Agent a
  documenté une position sur la façon dont sa sortie doit être rendue
  par une couche consommatrice (dashboard, adaptateur de gateway,
  écriture de fichiers, shell) et où un chemin de code enfreint cette
  position.

### 3.2 Hors périmètre

« Hors périmètre » signifie ici « pas une vulnérabilité de sécurité au
sens de cette politique ». Cela ne signifie pas « pas digne d'être
signalé ». Les améliorations des heuristiques intra-processus, les
idées de durcissement et les corrections d'UX sont les bienvenues sous
forme d'issues ou de pull requests classiques — le portail
d'approbation peut toujours attraper plus de motifs, la rédaction peut
toujours devenir plus fine, le comportement des adaptateurs peut
toujours être resserré. Ces éléments ne passent simplement pas par le
canal de divulgation privé et ne donnent pas lieu à des avis de
sécurité.

- **Les contournements des heuristiques intra-processus (§2.4)** —
  contournements des regex du portail d'approbation, contournements de
  la rédaction, contournements des motifs de Skills Guard, et
  signalements analogues visant de futures heuristiques. Ces composants
  ne sont pas des frontières ; les déjouer n'est pas une vulnérabilité
  au sens de cette politique.
- **L'injection de prompt en soi.** Amener le LLM à émettre une sortie
  inhabituelle — via du contenu injecté, une hallucination, des
  artefacts d'entraînement ou toute autre cause — n'est pas en soi une
  vulnérabilité. « J'ai réussi une injection de prompt » sans
  enchaînement vers un résultat du §3.1 n'est pas un signalement
  exploitable au sens de cette politique.
- **Les conséquences d'une posture d'isolation choisie.** Les
  signalements indiquant qu'un chemin de code opérant dans le périmètre
  de sa posture peut faire ce que cette posture permet ne sont pas des
  vulnérabilités. Exemples : des outils shell ou fichiers atteignant
  l'état de l'hôte sous le backend local ; des sous-processus
  d'exécution de code ou MCP atteignant l'état de l'hôte sous une
  isolation par backend de terminal qui ne sandboxe que le shell ; des
  signalements dont les préconditions exigent un accès en écriture
  préexistant à des fichiers de configuration ou d'identifiants
  appartenant à l'opérateur (ceux-ci sont déjà à l'intérieur de
  l'enveloppe de confiance).
- **Les réglages de dernier recours documentés.** Les compromis choisis
  par l'opérateur qui désactivent explicitement des protections :
  `--insecure` et les flags équivalents sur le dashboard ou d'autres
  composants, approbations désactivées, backend local en production,
  profils de développement qui contournent la sécurité de hermes-home,
  et similaires. Les signalements contre ces configurations ne sont pas
  des vulnérabilités — c'est précisément le rôle du flag.
- **Les skills et plugins issus de la communauté.** Les skills tierces
  (y compris le dépôt de skills de la communauté) et les plugins tiers
  relèvent de la surface de revue de l'opérateur, pas de la surface de
  confiance de Hermes Agent (§2.4, §2.5). Une skill ou un plugin qui
  fait quelque chose de malveillant est le mode de défaillance attendu
  d'un élément qui n'a pas été passé en revue, pas une vulnérabilité de
  Hermes Agent. Les bugs dans le chemin d'installation des skills ou
  des plugins de Hermes Agent qui empêchent l'opérateur de voir ce
  qu'il installe sont dans le périmètre au titre du §3.1.
- **L'exposition publique sans contrôles externes.** Exposer le gateway
  ou l'API à l'internet public sans authentification, VPN ni pare-feu.
- **Les restrictions de lecture/écriture au niveau des outils dans une
  posture où le shell est permis.** Si un chemin est atteignable via
  l'outil terminal, les signalements indiquant que d'autres outils de
  fichiers peuvent l'atteindre n'apportent rien.

---

## 4. Durcissement du déploiement

La décision de durcissement la plus importante est de faire
correspondre l'isolation (§2.2) au niveau de confiance du contenu que
l'agent va ingérer. Au-delà de cela :

- Exécutez l'agent en tant qu'utilisateur non root. L'image de
  conteneur fournie le fait par défaut.
- Conservez les identifiants dans le fichier d'identifiants de
  l'opérateur avec des permissions strictes, jamais dans la
  configuration principale, jamais sous contrôle de version. Sous
  OpenShell, utilisez le magasin Provider plutôt qu'un fichier
  d'identifiants sur disque.
- N'exposez pas le gateway ou l'API à l'internet public sans VPN,
  Tailscale ou protection par pare-feu. Sous OpenShell, utilisez la
  couche de politique réseau pour restreindre l'egress.
- Configurez une liste d'appelants autorisés pour chaque adaptateur
  exposé au réseau que vous activez (§2.6).
- Passez en revue les skills et plugins tiers avant installation
  (§2.4, §2.5). Pour les skills, cela signifie lire le Python et les
  scripts, pas seulement SKILL.md. Les rapports de Skills Guard et le
  journal d'audit d'installation constituent la surface de revue.
- Hermes Agent inclut des garde-fous de chaîne d'approvisionnement pour
  le lancement des serveurs MCP et pour les changements de dépendances
  / paquets embarqués en CI ; voir `CONTRIBUTING.md` pour les détails.

---

## 5. Divulgation

- **Fenêtre de divulgation coordonnée :** 90 jours à compter du
  signalement, ou jusqu'à la publication d'un correctif, selon la
  première échéance.
- **Canal :** le fil GHSA ou la correspondance par e-mail avec
  security@nousresearch.com.
- **Crédit :** les rapporteurs sont crédités dans les notes de version,
  sauf demande d'anonymat.
