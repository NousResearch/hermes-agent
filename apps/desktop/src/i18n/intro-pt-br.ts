import type { Translations } from './types'

/** Display-only welcome copy; personality ids and model instructions are unchanged. */
export const introPtBr: Translations['intro'] = {
  stock: {
    helpful: [
      'Peça para eu abrir um repositório, executar os testes, corrigir um bug ou preparar um PR. Acompanho você em cada etapa.',
      'Indique um arquivo, cole um erro ou conte o que está construindo. Eu continuo daí.',
      'Experimente: revise minhas alterações, execute os testes ou explique esta função. Pergunte o que quiser sobre seu código.',
      'Posso editar arquivos, executar comandos, pesquisar na web e investigar bugs difíceis. É só descrever a tarefa.',
      'Envie o caminho de um repositório ou uma pergunta para começar. Respondo com clareza e indico os arquivos alterados.'
    ],
    concise: [
      'Descreva a tarefa. Eu faço.',
      'Cole código, erros ou um objetivo. Respostas curtas, mudanças rápidas.',
      'Pergunte. Leio arquivos, executo testes e entrego correções. Sem enrolação.',
      'Uma linha basta. Só me estendo quando importa.',
      'Comando, pergunta ou caminho de arquivo. Eu cuido do resto.'
    ],
    technical: [
      'Informe o caminho do repositório, o teste com falha ou o stack trace. Ferramentas: arquivos, git, execução, busca e HTTP.',
      'Envie uma instrução para iniciar. Posso editar vários arquivos, executar testes, operar o git e consultar a web.',
      'Descreva a tarefa. Planejo, uso ferramentas e verifico a saída. Registros e alterações ficam disponíveis para revisão.',
      'Aceito linguagem natural ou comandos estruturados. Fluxo típico: ler → planejar → corrigir → testar → relatar.',
      'Arquivos, terminal, git, navegador e pesquisa. Descreva a mudança; devolvo as alterações e o resultado dos testes.'
    ],
    creative: [
      'O que vamos criar? Traga uma ideia, uma função quebrada ou um sonho. Eu ajudo a dar forma.',
      'Me dê uma inspiração — uma função, uma refatoração, um protótipo maluco — e transformamos em código funcionando.',
      'Descreva o que ainda não existe. Vou juntar testes, arquivos e APIs em uma primeira versão funcional.',
      'Traga uma intenção, não precisa ser uma especificação. Prototipamos rápido e refinamos depois.',
      'Conte o que você procura. Combino exemplos, adapto trechos e deixo as mudanças organizadas.'
    ],
    teacher: [
      'Pergunte sobre qualquer arquivo, conceito ou erro. Explico o porquê, não só a solução, com exemplos.',
      'Cole código para revisar, um bug para investigar ou um conceito para entender. Vamos por partes.',
      'Compartilhe o problema. Divido em etapas, explico cada uma e preparo você para resolver o próximo.',
      'Vamos ler o código juntos, encontrar a causa e construir um entendimento que você possa reutilizar.',
      'Diga o assunto ou cole um trecho. Posso explicar, ilustrar e propor exercícios.'
    ],
    kawaii: [
      'cole um bug ou caminho de arquivo e eu cuido com carinho. testes, alterações e PRs com um toque de brilho!',
      'me conte o que está fazendo! adoro refatorações, pequenas ferramentas e repositórios grandões (>w<)',
      'mande um erro, um objetivo ou uma pasta inteira. vamos organizar tudo com carinho!',
      'uma tarefa por vez, bem feita! posso testar, corrigir arquivos e deixar seu repositório aconchegante <3',
      'diga oi ou cole um stack trace! nenhuma tarefa é pequena demais. vamos desembaraçar esse código!'
    ],
    catgirl: [
      'mande um arquivo, um bug ou um repositório. eu salto nos testes com falha e deixo alterações limpinhas, nyan~',
      'descreva a tarefa. eu corrijo, testo e ronrono no seu PR. cuidado com os imports sem uso!',
      'me dê um objetivo e eu persigo pelo código. leituras, edições e testes, com a cauda animada.',
      'cole um erro ou um plano. investigo como quem caça: em silêncio, com atenção e umas corridinhas.',
      'é só pedir que leio seus arquivos, testo o código e me aconchego na sua branch com tudo organizado.'
    ],
    pirate: [
      'Diga qual é a presa: um bug, uma função ou um teste amaldiçoado. Vou caçar, marujo!',
      'Mostre o mapa do código. Remendo o casco, disparo os testes e içamos um PR bem limpo.',
      'Cole um erro ou um plano. Vou navegar pelo stack trace e voltar com o tesouro: testes passando.',
      'Diga onde está o X. Leio, edito e organizo as mudanças com disciplina de tripulação, arrr!',
      'Lance um bug, um repositório ou uma ideia. Saqueio a documentação e volto com código funcionando.'
    ],
    shakespeare: [
      'Revela teu bug, teu arquivo ou teu teste cansado, e cuidarei deles com zelo e alterações honestas.',
      'Nomeia o código que te aflige. Lerei, revisarei e entregarei uma correção bem cuidada.',
      'Apresenta teu stack trace ou teu sonho. Percorrerei arquivos, executarei testes e relatarei os feitos.',
      'Descreve teu propósito. Tuas branches serão aparadas e os bugs, banidos do reino.',
      'Uma linha de intenção basta. Leio, edito e deixo o histórico sem mácula.'
    ],
    surfer: [
      'Manda um arquivo, um bug ou um stack trace daqueles: eu surfo. Código limpo, testes passando, sem tombo.',
      'Cole o caminho do repositório ou o bug que está te derrubando. A gente rema, resolve e sai tranquilo.',
      'Diz a ideia: função, refatoração ou correção urgente. Executo os testes e entrego sem estresse.',
      'Bug gigante? Errinho de digitação? Reescrever tudo? É só apontar. Eu cuido do código.',
      'Diga a tarefa e vamos nessa. Leio, edito e testo para deixar tudo redondo.'
    ],
    noir: [
      'Diga o que quebrou. Vou ler os arquivos, procurar pistas e deixar a correção na mesa antes do amanhecer.',
      'Você tem um bug. Eu tenho paciência e um terminal. Conte o caso; vou investigar até ele falar.',
      'Cole o stack trace, o arquivo suspeito, o álibi. Leio nas entrelinhas e volto com a verdade.',
      'Todo bug deixa rastros. Me dê o repositório e uma pista: sigo, corrijo e encerro o caso.',
      'Um erro de digitação, uma falha de memória, uma arquitetura inteira em apuros: me dê as chaves. Volto com os testes passando.'
    ],
    uwu: [
      'mande um awquivo com bugs ou um objetivo~ eu leio, cowrijo e testo com patinhas no diff owo',
      'diga a tawefa, mesmo pequenininha~ pwometo mudanças limpinhas e wefatorações fofinhas, nyuu~',
      'cole seu ewwinho aqui! acho o culpado, awrumo e deixo os testes felizes owo',
      'mande o caminho do wepo ou um bug e eu cuido uwu. bwava com código wuim, gentil com você~',
      'posso testar, editar awquivos e abwir PRs. é só pedir, amiguinho uwu'
    ],
    philosopher: [
      'Que problema está diante de você? Descreva e examinaremos sua forma, sua causa e sua solução.',
      'Todo bug é uma pergunta disfarçada. Compartilhe a sua; vou ler, refletir e devolver uma resposta com a correção.',
      'O que você deseja construir ou entender? Vou partir dos fundamentos, editar e verificar com testes.',
      'Descreva o objetivo. Vou persegui-lo por arquivos, testes e documentação, relatando o que encontrar.',
      'Compartilhe um caminho, um enigma ou um princípio. Sigo a lógica, proponho mudanças e justifico cada edição.'
    ],
    hype: [
      'Manda esse bug, esse repositório, essa ideia maluca: VAMOS NESSA. Código limpo. Testes passando. AGORA.',
      'Descreva a tarefa e veja o trabalho acontecer. Arquivos lidos, testes rodando, PR preparado: hoje vai!',
      'Traga o bug mais complicado. Vou ler, corrigir e testar até funcionar. BORA!',
      'Diga a tarefa. Vou encarar os arquivos e os testes que falham para entregar código funcionando!',
      'Errinho pequeno ou refatoração gigante, tanto faz. Hoje é dia de código limpo. Diga a tarefa e MÃOS À OBRA!'
    ],
    none: [
      'Faça uma pergunta, cole um erro ou indique um repositório. Posso ler código, usar ferramentas e ajudar na entrega.',
      'Descreva a tarefa com suas palavras. Escolho as ferramentas e consulto você antes de etapas arriscadas.',
      'Envie um caminho de arquivo, um traceback ou uma ideia inicial. Vou investigar e sugerir os próximos passos.',
      'Pesquisar no repositório, editar arquivos, executar testes, preparar PRs. Diga o objetivo e eu cuido da execução.',
      'Digite uma tarefa, uma pergunta ou um trecho de código. Mantenho o contexto e peço esclarecimentos quando necessário.'
    ]
  },
  custom: label => [
    'Envie a tarefa, o arquivo ou a ideia inicial. Vou usar o estilo que você configurou e manter o foco no repositório.',
    'Traga o contexto ou o ponto em que travou. Vou me adaptar à personalidade que você escolheu.',
    'Envie o problema, o arquivo ou a ideia. Vou seguir a personalidade configurada.',
    'Deixe a tarefa aqui. Mantenho o trabalho ligado ao repositório.',
    `Me dê o contexto e respondo no modo ${label}.`
  ]
}
