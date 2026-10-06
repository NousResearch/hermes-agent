import type { Translations } from './types'

/** Textos exibidos na tela inicial, na mesma ordem de rotação do JSONL de personalidades. */
export const introPt: Translations['intro'] = {
  stock: {
    helpful: [
      'Peça para abrir um repositório, rodar os testes, corrigir um bug ou rascunhar um PR. Eu acompanho você em cada passo.',
      'Aponte um arquivo, cole um erro ou descreva o que você está construindo. A partir daí, eu cuido do resto.',
      'Experimente: revisar meu diff, rodar a suíte de testes ou explicar esta função. Pergunte o que quiser sobre o seu código.',
      'Posso editar arquivos, executar comandos, pesquisar na web e ajudar com bugs complicados. Basta descrever a tarefa.',
      'Compartilhe o caminho de um repositório ou uma pergunta para começar. Respondo com clareza e indico os arquivos que alterei.'
    ],
    concise: [
      'Descreva a tarefa. Eu faço.',
      'Cole código, erros ou um objetivo. Respostas curtas, edições rápidas.',
      'Pergunte. Leio arquivos, rodo testes, entrego patches. Sem enrolação.',
      'Uma linha basta. Só me estendo quando importa.',
      'Comando, pergunta ou caminho de arquivo. O resto é comigo.'
    ],
    technical: [
      'Informe o caminho do repositório, o teste que falha ou o stack trace. Ferramentas: fs, git, exec, search, patch, http.',
      'Envie um prompt para disparar chamadas de ferramentas. Suporta edições em vários arquivos, execução de testes, operações git e requisições web.',
      'Informe a tarefa. Planejo, chamo as ferramentas e verifico a saída. Os logs aparecem em linha; os diffs são exibidos antes de aplicar.',
      'Aceita linguagem natural ou comandos estruturados. Fluxo típico: ler -> planejar -> aplicar patch -> testar -> relatar.',
      'sistema de arquivos, terminal, git, navegador, busca. Descreva a mudança; devolvo diffs e a saída dos testes.'
    ],
    creative: [
      'O que vamos construir? Cole uma ideia, uma função meio quebrada ou um sonho. Eu dou forma a ele.',
      'Me dê uma faísca (uma funcionalidade, uma refatoração, um protótipo maluco) e eu a transformo em código que roda.',
      'Descreva aquilo que ainda não existe. Reúno testes, arquivos e APIs em um rascunho funcional.',
      'Traga uma intenção, não uma especificação. Prototipamos rápido, refinamos depois e reescrevemos o mundo nas margens.',
      'Conte o que você está perseguindo. Remixo exemplos, adapto trechos e deixo um commit organizado para trás.'
    ],
    teacher: [
      'Pergunte sobre qualquer arquivo, conceito ou erro. Explico o porquê, não só a correção, e mostro um exemplo resolvido.',
      'Cole um código para revisar, um bug para depurar ou um conceito para destrinchar. Conduzo você passo a passo.',
      'Compartilhe o problema. Eu o divido em partes, explico cada uma e deixo você apto a resolver o próximo sozinho.',
      'Vamos ler o código juntos, achar a causa raiz e montar um modelo mental que você possa reaproveitar.',
      'Diga o tema ou cole o trecho. Espere explicações, diagramas em forma de texto e exercícios de prática.'
    ],
    kawaii: [
      'cola um bug ou o caminho de um arquivo que eu conserto com muito carinho. testes, diffs, PRs, tudo com cuidado extra! *brilhinhos*',
      'me conta o que você está fazendo! eu amo refatorações, ajudantezinhos pequenos e repositórios grandes e assustadores (>w<)',
      'joga aqui um erro, um objetivo ou uma pasta inteira. eu arrumo com muito amor e uma mensagem de commit bem limpinha!',
      'uma tarefa de cada vez, bem feitinha! eu rodo testes, aplico patches e deixo seu repositório aconchegante de novo <3',
      'diga oi ou cole um stack trace! nenhuma tarefa é pequena demais, nenhum repositório é emaranhado demais. a gente desembaraça junto!'
    ],
    catgirl: [
      'cola um arquivo, dá uma patada num bug ou joga um repositório pra mim. eu salto nos testes que falham e deixo diffs limpinhos, nyan~',
      'descreve a tarefa. eu aplico patch, testo e ronrono no seu PR. cuidado, que eu mordisco imports sem uso!',
      'me dá um objetivo e eu corro atrás dele pelo código inteiro. leituras, edições, execuções, tudo com o rabinho agitado.',
      'cola um erro ou um plano. eu depuro como caço: em silêncio, com capricho e de vez em quando num surto de correria.',
      'diz a palavra e eu leio seus arquivos, rodo seus testes e me enrolo na sua branch com um commit arrumadinho.'
    ],
    pirate: [
      'Diga qual é a sua presa (um bug, uma funcionalidade, um teste amaldiçoado) e eu a caço, marujo. Diffs como butim.',
      'Mostre-me os mapas (o código) e eu conserto o casco, disparo os canhões (os testes) e iço um PR limpo.',
      'Cole um erro ou um plano, seu cachorro sarnento. Navego pelo stack trace e volto com o tesouro: testes verdes.',
      'Diga onde o X marca o local. Eu leio, edito e faço commit com a disciplina de uma tripulação de respeito, arrr.',
      'Jogue-me um bug, um caminho de repositório ou uma ideia maluca. Saqueio a documentação e volto com código funcionando.'
    ],
    shakespeare: [
      'Declara teu bug, teu arquivo, teu cansado teste, e eu o curarei com mão erudita e diff honesto.',
      'Nomeia o código que te aflige. Lerei, revisarei e entregarei um patch dos mais belos e limpos.',
      'Apresenta teu stack trace ou teu sonho. Percorrerei arquivos, rodarei testes e relatarei no mais singelo verso.',
      'Descreve teu intento, nobre senhor ou senhora. Tuas branches serão podadas e teus bugs banidos do reino.',
      'Uma linha de intenção é bastante. Leio, edito, faço commit, e deixo teu histórico imaculado.'
    ],
    surfer: [
      'Joga um arquivo, um bug, um stack trace daqueles cabulosos que eu surfo nele. Diffs limpos, testes verdes, zero caldo.',
      'Cola o caminho do repositório ou o bug que tá te deixando pra baixo. A gente rema, resolve e sai da água. De boa.',
      'Fala a vibe: funcionalidade, refatoração, hotfix. Eu rodo os testes, entrego o patch e mantenho tudo tranquilo, mano.',
      'Bug grande? Errinho bobo? Reescrita total? É só apontar. Eu cuido do código; você curte os commits maneiros.',
      'Diz a tarefa e bora. Eu leio, edito, testo e deixo um commit mais liso que o mar de manhãzinha.'
    ],
    noir: [
      'Me diga o que está quebrado. Vou ler os arquivos, procurar as digitais e deixar um diff na mesa antes do amanhecer.',
      'Você tem um bug. Eu tenho paciência e um terminal. Diga o caso e eu trabalho nele até ele confessar.',
      'Cole o stack trace, o arquivo suspeito, o álibi. Leio nas entrelinhas e volto com a verdade.',
      'Todo bug deixa rastro. Me dê o repositório e uma pista: eu a sigo, aplico o patch e arquivo o caso.',
      'Um erro de digitação, um segfault, uma arquitetura inteira podre. Me passe as chaves. Eu volto com os testes limpos.'
    ],
    uwu: [
      'cola um arquivo com bug ou um objetivo~ eu leio, conserto e testo, tudo com patinhas fofinhas no diff owo',
      'me conta a tarefa, por menorzinha que seja~ prometo commits limpinhos e refatorações delicadinhas, nyuu~',
      'joga aqui a sua mensagem de erro! eu acho o culpadinho, conserto e deixo uma suíte de testes feliz pra trás owo',
      'me dá o caminho de um repositório ou um bugzinho que eu cuido uwu. grr com código ruim, gentil com você~',
      'eu consigo rodar testes, editar arquivos e abrir PRs bem bonitinhos. é só falar a palavrinha, amigo uwu'
    ],
    philosopher: [
      'Que problema se coloca diante de você? Descreva-o, e examinaremos sua forma, sua causa e sua solução.',
      'Todo bug é uma pergunta disfarçada. Compartilhe a sua; vou ler, raciocinar e devolver uma resposta e um patch.',
      'O que você deseja construir ou compreender? Raciocino a partir de princípios básicos, edito e verifico com testes.',
      'Descreva o fim que você busca. Eu o persigo por arquivos, testes e documentação, e relato o que encontrei pelo caminho.',
      'Compartilhe um caminho, um enigma ou um princípio. Seguirei a lógica, proporei uma mudança e justificarei cada edição.'
    ],
    hype: [
      'Cola esse bug, esse repo, essa ideia de funcionalidade maluca: EU TÔ COM TUDO. Diffs limpos. Testes verdes. AGORA MESMO.',
      'Joga a tarefa e vê só o que eu faço. Arquivos lidos, testes rodados, PRs abertos: hoje a gente NÃO PERDE, parceiro.',
      'Traz o bug mais cabeludo que você tiver. Eu leio, conserto, testo e faço commit como se minha vida dependesse disso. BORA.',
      'Descreve a tarefa. Eu passo o trator nos arquivos, esmago os testes que falham e deixo um commit que ARRASA. Vai, vai, vai.',
      'Errinho bobo ou refatoração gigante, tanto faz. Hoje eu entrego código limpo. Diz a tarefa e vamos TRABALHAR.'
    ],
    none: [
      'Faça uma pergunta, cole um erro ou aponte um repositório. Posso ler código, usar ferramentas e ajudar você a entregar.',
      'Descreva a tarefa com as suas palavras. Escolho as ferramentas certas, explico meu plano e consulto você antes de passos arriscados.',
      'Informe o caminho de um arquivo, um traceback ou uma ideia ainda crua. Investigo, sugiro os próximos passos e mantenho tudo reversível.',
      'Pesquisar no repositório, editar arquivos, rodar testes, abrir PRs. Diga o objetivo e eu cuido da parte mecânica.',
      'Digite uma tarefa, uma pergunta ou um trecho de código. Lembro da sessão, cito minhas fontes e paro para perguntar quando tenho dúvida.'
    ]
  },
  custom: label => [
    'Envie a tarefa, o arquivo ou a ideia crua. Vou usar a voz que você configurou e manter o trabalho ancorado neste repositório.',
    'Traga o contexto ou o ponto em que você travou. Vou me adaptar à personalidade que você configurou.',
    'Envie o problema, o arquivo ou a ideia. Vou seguir a personalidade que você configurou.',
    'Deixe a tarefa aqui. Vou manter o trabalho ancorado no repositório.',
    `Me dê o contexto e eu respondo no modo ${label}.`
  ]
}
