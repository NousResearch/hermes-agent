import type { Translations } from './types'

/** Display-only translations, in the stock JSONL's per-personality rotation order. */
export const introFr: Translations['intro'] = {
  stock: {
    helpful: [
      'Demandez-moi d’ouvrir un dépôt, de lancer les tests, de corriger un bug ou de rédiger une PR. Je vous guide étape par étape.',
      'Indiquez-moi un fichier, collez une erreur ou décrivez ce que vous construisez. Je m’occupe du reste.',
      'Essayez : relis mon diff, lance la suite de tests ou explique cette fonction. Posez n’importe quelle question sur votre code.',
      'Je peux modifier des fichiers, exécuter des commandes, chercher sur le web et vous aider sur les bugs retors. Décrivez simplement la tâche.',
      'Partagez un chemin de dépôt ou une question pour commencer. Je réponds clairement et renvoie vers les fichiers que je modifie.'
    ],
    concise: [
      'Décrivez la tâche. Je m’en charge.',
      'Collez du code, une erreur ou un objectif. Réponses courtes, modifications rapides.',
      'Demandez. Je lis les fichiers, lance les tests, livre les correctifs. Sans blabla.',
      'Une ligne suffit. Je développe seulement quand c’est utile.',
      'Commande, question ou chemin de fichier. Je gère le reste.'
    ],
    technical: [
      'Fournissez le chemin du dépôt, le test en échec ou la stack trace. Outils : fs, git, exec, search, patch, http.',
      'Envoyez un prompt pour déclencher des appels d’outils. Prend en charge les modifications multi-fichiers, les tests, les opérations git et les requêtes web.',
      'Saisissez la tâche. Je planifie, appelle les outils, vérifie la sortie. Journaux en direct ; diffs fournis avant application.',
      'Accepte le langage naturel ou des commandes structurées. Flux type : lecture -> plan -> patch -> test -> rapport.',
      'système de fichiers, terminal, git, navigateur, recherche. Décrivez la modification ; je renvoie les diffs et la sortie des tests.'
    ],
    creative: [
      'Que construisons-nous ? Collez une idée, une fonction à moitié cassée ou un rêve. Je lui donnerai forme.',
      'Donnez-moi une étincelle — une fonctionnalité, un refactoring, un prototype fou — et j’en ferai du code qui tourne.',
      'Décrivez ce qui n’existe pas encore. J’assemble tests, fichiers et API en un premier jet fonctionnel.',
      'Apportez une intention, pas un cahier des charges. Prototypons vite, peaufinons ensuite et réécrivons le monde dans la marge.',
      'Dites-moi ce que vous poursuivez. Je remixe des exemples, adapte des extraits et laisse un commit soigné derrière moi.'
    ],
    teacher: [
      'Posez une question sur un fichier, un concept ou une erreur. J’explique le pourquoi, pas seulement le correctif, avec un exemple détaillé.',
      'Collez du code à relire, un bug à traquer ou un concept à décortiquer. Je vous guide pas à pas.',
      'Partagez le problème. Je le découpe, explique chaque partie et vous laisse capable de résoudre le suivant seul.',
      'Nous lirons le code ensemble, trouverons la cause racine et bâtirons un modèle mental réutilisable.',
      'Nommez le sujet ou collez l’extrait. Au programme : explications, schémas en prose et exercices.'
    ],
    kawaii: [
      'colle un bug ou un chemin de fichier et je le répare tout en douceur. tests, diffs, PR — avec encore plus d’attention ! *paillettes*',
      'dis-moi ce que tu fabriques ! j’adore les refactorings, les petits utilitaires et les gros dépôts effrayants (>w<)',
      'envoie une erreur, un objectif ou tout un dossier. je range tout avec plein d’amour et un message de commit tout propre !',
      'une tâche à la fois, bien faite ! je peux lancer les tests, corriger des fichiers et rendre ton dépôt douillet à nouveau <3',
      'dis bonjour ou colle une stack trace ! aucune tâche trop petite, aucun dépôt trop emmêlé. on démêle tout ensemble !'
    ],
    catgirl: [
      'colle un fichier, donne un coup de patte à un bug ou lance-moi un dépôt. je bondis sur les tests en échec et laisse des diffs propres, nyan~',
      'décris la tâche. je corrige, je teste et je ronronne sur ta PR. attention — je mordille les imports inutilisés !',
      'donne-moi un objectif et je le poursuis dans tout le code. lecture, modifs, exécution — la queue frétillante.',
      'colle une erreur ou un plan. je débogue comme je chasse : en silence, à fond, avec un petit sprint de temps en temps.',
      'dis un mot et je lis tes fichiers, lance tes tests et me roule en boule dans ta branche avec un commit bien rangé.'
    ],
    pirate: [
      'Nomme ta proie — un bug, une fonctionnalité, un test maudit — et je la traquerai, moussaillon. Des diffs pour butin.',
      'Montre-moi les cartes (le code) et je colmaterai la coque, tirerai les canons (les tests), hisserai une PR propre.',
      'Colle une erreur ou un plan, vieux loup de mer. Je navigue dans la stack trace et rapporte le trésor : des tests au vert.',
      'Dis-moi où la croix marque l’endroit. Je lis, modifie et commite avec la discipline d’un vrai équipage, arrr.',
      'Lance-moi un bug, un chemin de dépôt ou une idée folle. Je pille la doc et reviens avec du code qui marche.'
    ],
    shakespeare: [
      'Conte-moi ton bug, ton fichier, ton test las, et je le réparerai d’une main savante et d’un diff honnête.',
      'Nomme le code qui te tourmente. Je lirai, réviserai et rendrai un correctif des plus beaux et des plus nets.',
      'Présente ta stack trace ou ton rêve. Je parcourrai les fichiers, lancerai les tests et rendrai compte en vers fort simples.',
      'Décris ton dessein, noble dame ou gentilhomme. Tes branches seront taillées, tes bugs bannis du royaume.',
      'Une ligne d’intention suffit. Je lis, je modifie, je commite — et laisse ton histoire sans tache.'
    ],
    surfer: [
      'Balance un fichier, un bug, une stack trace bien chaude — je surfe dessus. Diffs propres, tests au vert, zéro gamelle.',
      'Colle le chemin de ton dépôt ou le bug qui te plombe. On rame, on corrige, on ressort. Tranquille.',
      'Dis-moi l’ambiance : fonctionnalité, refactoring, hotfix. Je lance les tests, livre le patch et je reste zen, mec.',
      'Gros bug ? Petite coquille ? Réécriture complète ? Montre-moi. Je m’occupe du code ; toi, tu profites des commits.',
      'Donne la tâche et c’est parti. Je lis, modifie, teste et laisse un commit plus lisse qu’une session à l’aube.'
    ],
    noir: [
      'Dites-moi ce qui est cassé. Je lirai les fichiers, relèverai les empreintes et laisserai un diff sur le bureau d’ici le matin.',
      'Vous avez un bug. J’ai de la patience et un terminal. Donnez-moi l’affaire, je la travaillerai jusqu’à ce qu’elle parle.',
      'Collez la stack trace, le fichier suspect, l’alibi. Je lis entre les lignes et reviens avec la vérité.',
      'Chaque bug laisse une piste. Donnez-moi le dépôt et un indice — je la suivrai, corrigerai et classerai le dossier.',
      'Une coquille, un segfault, toute une architecture pourrie — donnez-moi les clés. Je reviendrai avec des tests propres.'
    ],
    uwu: [
      'cowwe un fichiew buggé ou un objectif~ je vais wiwe, cowwiger et testew, avec des petites empweintes suw we diff owo',
      'dis-moi wa tâche, même toute petite~ je pwomets des commits pwopwes et des wefactowings tout doux, nyuu~',
      'envoie ton message d’ewweuw ici ! je twouve we coupabwe, je wépawe et je waisse des tests tout contents owo',
      'donne-moi un chemin de dépôt ou un petit bug et je m’en occupe uwu. gwww contwe we mauvais code, gentiw avec toi~',
      'je peux wancew des tests, modifiew des fichiews et ouvwiw des PR à wewiwe. dis juste we mot, copain uwu'
    ],
    philosopher: [
      'Quel problème se tient devant vous ? Décrivez-le, et nous en examinerons la forme, la cause et la solution.',
      'Chaque bug est une question déguisée. Partagez la vôtre ; je lirai, raisonnerai et rendrai une réponse — et un correctif.',
      'Que souhaitez-vous construire, ou comprendre ? Je raisonnerai à partir des premiers principes, modifierai et vérifierai par les tests.',
      'Décrivez la fin que vous recherchez. Je la poursuis à travers fichiers, tests et docs, et rends compte de ce que je trouve en chemin.',
      'Partagez un chemin, une énigme ou un principe. Je suivrai la logique, proposerai une modification et justifierai chaque changement.'
    ],
    hype: [
      'Balance ce bug, ce dépôt, cette idée de fonctionnalité de fou — JE SUIS À FOND. Diffs propres. Tests au vert. TOUT DE SUITE.',
      'Donne ta tâche et regarde-moi envoyer. Fichiers lus, tests lancés, PR ouvertes — on ne perd PAS aujourd’hui, l’ami.',
      'Amène le bug le plus coriace que tu as. Je lis, corrige, teste et commite comme si ma vie en dépendait. C’EST PARTI.',
      'Décris la tâche. Je fonce dans les fichiers, j’écrase les tests en échec et je laisse un commit qui DÉCHIRE. Go go go.',
      'Petite coquille ou gros refactoring — peu importe. Je livre du code propre aujourd’hui. Donne la tâche et au BOULOT.'
    ],
    none: [
      'Posez une question, collez une erreur ou indiquez-moi un dépôt. Je peux lire du code, utiliser des outils et vous aider à livrer.',
      'Décrivez la tâche avec vos mots. Je choisis les bons outils, explique mon plan et vous consulte avant les étapes risquées.',
      'Indiquez un chemin de fichier, une traceback ou une idée brute. J’enquête, propose les étapes suivantes et garde tout réversible.',
      'Cherchez dans le dépôt, modifiez des fichiers, lancez les tests, ouvrez des PR. Donnez-moi l’objectif, je gère la partie mécanique.',
      'Saisissez une tâche, une question ou un extrait. Je garde en mémoire la session, cite mes sources et m’arrête pour demander en cas de doute.'
    ]
  },
  custom: label => [
    'Envoyez une tâche, un fichier ou une idée. Je suivrai la voix configurée et resterai fidèle à la réalité du dépôt.',
    'Donnez-moi le contexte et l’endroit où vous bloquez. Je m’adapte à la personnalité configurée.',
    'Envoyez un problème, un fichier ou une idée. Je suivrai la personnalité configurée.',
    'Déposez la tâche ici. Je travaillerai en restant fidèle à la réalité du dépôt.',
    `Donnez-moi le contexte. Je réponds en mode ${label}.`
  ]
}
