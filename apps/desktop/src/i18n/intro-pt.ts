import type { Translations } from './types'

/** Display-only translations, in the stock JSONL's per-personality rotation order. */
export const introPt: Translations['intro'] = {
  stock: {
    helpful: [
      'Peça que eu abra um repositório, rode os testes, corrige um bug ou redija um PR. Eu acompanho o passo a passo com você.',
      'Aponte um arquivo, cole um erro ou descreva o que você está construindo. Eu cuido do resto a partir daí.',
      'Experimente: revise meu diff, rode a suíte de testes ou explique esta função. Pergunte o que quiser sobre o seu código.',
      'Posso editar arquivos, executar comandos, buscar na web e ajudar com bugs cabeludos. É só descrever a tarefa.',
      'Compartilhe o caminho de um repositório ou uma pergunta para começar. Respondo com clareza e referencio os arquivos que toco.'
    ],
    concise: [
      'Descreva a tarefa. Eu faço.',
      'Cole código, erros ou um objetivo. Respostas curtas, mudanças rápidas.',
      'Pergunte. Eu leio arquivos, rodo testes, entrego patches. Sem enrolação.',
      'Uma linha basta. Só me estendo quando importa.',
      'Comando, pergunta ou caminho de arquivo. Eu cuido do resto.'
    ],
    technical: [
      'Indique o caminho do repositório, o teste que falha ou o stack trace. Ferramentas: fs, git, exec, search, patch, http.',
      'Envie um prompt para disparar chamadas de ferramentas. Aceito edições em vários arquivos, execução de testes, operações de git e consultas web.',
      'Descreva a tarefa. Eu planejo, chamo as ferramentas e verifico a saída. Os logs aparecem em linha; os diffs voltam antes de aplicar.',
      'Aceito linguagem natural ou comandos estruturados. Fluxo típico: ler -> planejar -> aplicar patch -> testar -> reportar.',
      'Sistema de arquivos, terminal, git, navegador, busca. Descreva a mudança; eu devolvo diffs e a saída dos testes.'
    ],
    creative: [
      'O que vamos construir? Cole uma ideia, uma função meia-bomba ou um sonho. Eu dou forma a isso.',
      'Me dê uma faísca (uma função, uma refatoração, um protótipo maluco) e eu transformo em código que funciona.',
      'Descreva o que ainda não existe. Eu reúno testes, arquivos e API num rascunho funcional.',
      'Traga uma intenção, não uma especificação. Prototipamos rápido, polimos depois e reescrevemos o mundo nas margens.',
      'Me conte o que você persegue. Eu remixo exemplos, adapto fragmentos e deixo um commit bem arrumado.'
    ],
    teacher: [
      'Pergunte sobre qualquer arquivo, conceito ou erro. Eu explico o porquê, não só a solução, e mostro um exemplo resolvido.',
      'Cole código para revisar, um bug para depurar ou um conceito para destrinchar. Eu guio passo a passo.',
      'Compartilhe o problema. Eu divido em partes, explico cada uma e te deixo pronto para resolver a próxima sozinho.',
      'Vamos ler o código juntos, achar a causa raiz e construir um modelo mental que você reutiliza.',
      'Diga o tema ou cole o trecho. Terá explicações, diagramas em prosa e exercícios de prática.'
    ],
    kawaii: [
      'cole um bug ou um caminho de arquivo que eu arrumo com muito cuidado. testes, diffs, PR, tudo com carinho extra! *brilhinhos*',
      'me conta o que você tá fazendo! eu amo refatorações, utilidades pequenininhas e repos grandes e assustadores (>w<)',
      'solta um erro, um objetivo ou uma pasta inteira. eu organizo tudo com muito amor e uma mensagem de commit bem limpinha!',
      'uma tarefa por vez, bem feita! posso rodar testes, aplicar patches e deixar seu repo aconchegante de novo <3',
      'dá um oi ou cola um stack trace! nenhuma tarefa é pequena demais e nenhum repo é enrolado demais. a gente desenrola junto!'
    ],
    catgirl: [
      'cola um arquivo, dá uma patadinha num bug ou me joga um repo. eu pulo em cima dos testes que falharam e deixo diffs limpos, nyan~',
      'descreve a tarefa. eu patcheo, testo e ronrono no seu PR. cuidado, que eu mordisco imports sem uso!',
      'me dá um objetivo que eu persigo pelo código inteiro. leituras, edições, execuções, com o rabinho inquieto.',
      'cola um erro ou um plano. eu depuro como caça: em silêncio, a fundo e com algum acesso de corrida.',
      'diga a palavra que eu leio seus arquivos, rodo seus testes e me enrolo na sua branch com um commit arrumadinho.'
    ],
    pirate: [
      'Nomeie sua presa (um bug, uma função, um teste maldito) que eu caço, grumete. Diffs como espólio.',
      'Mostre as cartas (o código) que eu remendo o casco, disparo os canhões (os testes) e içio um PR limpo.',
      'Cole um erro ou um plano, cão escabroso. Navego o stack trace e volto com o tesouro: testes no verde.',
      'Diga onde o X marca. Eu leio, edito e committo com a disciplina de uma tripulação de verdade, arrr.',
      'Jogue um bug, o caminho de um repo ou uma ideia louca. Eu saqueio a documentação e volto com código que funciona.'
    ],
    shakespeare: [
      'Pronuncia tua tarefa, gentil criador: um bug a purgar, um teste a redimir, um PR a forjar.',
      'Cola o erro como quem confessa um segredo; eu, fiel escudeiro, trarei o diff redentor.',
      'Aponta o ato e a cena (arquivo e função) e assistiremos juntos o drama do código se resolver.',
      'Dize o que almejas, e com pena e patch eu reescreverei os versos que hoje tortos soam.',
      'Traz um problema ou um desejo; eu o transformo em verso commitado, limpo e eterno.'
    ],
    surfer: [
      'Manda a vibe, mano: um bug, um repo, uma ideia. Eu pego essa onda e volto com o diff.',
      'Cola o erro aí que eu inspecciono a maré do código e acho o pico do problema.',
      'Relaxa que eu leio os arquivos, rodo os testes e volto com tudo no verde, estilo limpo.',
      'Descreve a tarefinha. Remo fundo, mas chego com patch e teste bem docado.',
      'Só aponta o caminho do repo, cara. O resto é só cair na água e codar.'
    ],
    noir: [
      'Cola o erro. Toda cidade tem um bug corrupto; eu tenho um trench coat e bastante tempo.',
      'Descreva o caso. Eu interrogo os arquivos, sigo os imports suspeitos e volto com a verdade.',
      'Aponte o arquivo e a hora do crime. Nenhum stack trace escapa desta cidade.',
      'Um bug entrou no meu escritório pedindo ajuda. Diga onde ele mora e eu cuido do resto.',
      'Traga o mistério. Eu leio os logs como confissões e o diff fecha o caso.'
    ],
    uwu: [
      'c-cola um erro ou um caminho de awquivo que eu awwumo direitinho, uwu',
      'd-descreve a tawefa que eu pwanejo, testo e entwego com cawinho',
      's-solta um bug aqui que eu twaco dele com muitos testes e poupa pwomessa',
      'm-manda o wepo que eu wenho os awquivos e deixo tudo awcadinho',
      'd-diz o objetivo que eu pewseguo com o wabinho tremendo de empowgação'
    ],
    philosopher: [
      'Compartilhe um caminho, um enigma ou um princípio. Sigo a lógica, proponho uma mudança e justifico cada edição.',
      'O que é um bug senão uma suposição traída? Traga o erro; examinamos juntos suas premissas.',
      'Descreva o sistema e sua falha. Eu divido o problema até o axioma e reconstruo a partir dele.',
      'Pergunte pelo porquê do código, não apenas pelo como. Eu respondo com referências e diffs.',
      'Traga a dúvida. Methodicus: duvido do bug, testo a hipótese e só então commito a verdade.'
    ],
    hype: [
      'Cola esse bug, esse repo, essa ideia de função MALUCA: TÔ BOMBAADO. Diffs limpos. Testes no verde. AGORA.',
      'Solta sua tarefa e me vê DAR TUDO DE MIM. Arquivos lidos, testes rodados, PRs abertos: hoje NÃO perdemos, parça.',
      'Traz o bug mais retorcido que você tem. Leio, patcheo, testo e commito como se não houvesse amanhã. BORA.',
      'Descreve a tarefa. Eu arraso com os arquivos, esmago os testes vermelhos e deixo um commit QUE DETONA. Vamo, vamo, vamo.',
      'Errata minúscula ou refatoração gigante, tanto faz. Hoje eu entrego código limpo. Diz a tarefa e AO TRABALHO.'
    ],
    none: [
      'Faça uma pergunta, cole um erro ou aponte um repositório. Eu leio código, rodo ferramentas e ajudo a entregar.',
      'Descreva a tarefa com suas palavras. Escolho as ferramentas certas, explico o plano e te consulto antes dos passos arriscados.',
      'Solte um caminho de arquivo, um traceback ou uma ideia bruta. Investigo, sugiro os próximos passos e mantenho tudo reversível.',
      'Busque no repositório, edite arquivos, rode testes, abra PR. Diga o objetivo que eu cuido da parte mecânica.',
      'Escreva uma tarefa, uma pergunta ou um trecho. Eu lembro da sessão, cito minhas fontes e paro para perguntar quando tenho dúvida.'
    ]
  },
  custom: label => [
    'Manda a tarefa, o arquivo ou a ideia bruta. Uso a voz que você configurou e mantenho o trabalho ancorado neste repositório.',
    'Traga o contexto ou a parte onde você travou. Eu me adapto à personalidade que você configurou.',
    'Envie o problema, o arquivo ou a ideia. Sigo a personalidade que você configurou.',
    'Deixe a tarefa aqui. Eu mantenho o trabalho ancorado no repositório.',
    `Me dê o contexto que eu respondo no modo ${label}.`
  ]
}
