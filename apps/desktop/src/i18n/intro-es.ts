import type { Translations } from './types'

/** Display-only translations, in the stock JSONL's per-personality rotation order. */
export const introEs: Translations['intro'] = {
  stock: {
    helpful: [
      'Pídeme que abra un repositorio, ejecute las pruebas, corrija un bug o redacte un PR. Te acompaño paso a paso.',
      'Indícame un archivo, pega un error o describe lo que estás construyendo. Yo me encargo desde ahí.',
      'Prueba: revisa mi diff, ejecuta la suite de pruebas o explica esta función. Pregunta lo que quieras sobre tu código.',
      'Puedo editar archivos, ejecutar comandos, buscar en la web y ayudarte con bugs complicados. Solo describe la tarea.',
      'Comparte la ruta de un repositorio o una pregunta para empezar. Respondo con claridad y enlazo los archivos que toco.'
    ],
    concise: [
      'Describe la tarea. Yo la hago.',
      'Pega código, errores o un objetivo. Respuestas cortas, cambios rápidos.',
      'Pregunta. Leo archivos, ejecuto pruebas, entrego parches. Sin relleno.',
      'Una línea basta. Solo me extiendo cuando importa.',
      'Comando, pregunta o ruta de archivo. Yo me encargo del resto.'
    ],
    technical: [
      'Indica la ruta del repositorio, la prueba que falla o el stack trace. Herramientas: fs, git, exec, search, patch, http.',
      'Envía un prompt para lanzar llamadas a herramientas. Admite ediciones en varios archivos, ejecución de pruebas, operaciones de git y consultas web.',
      'Introduce la tarea. Planifico, llamo a herramientas y verifico la salida. Los registros se muestran en línea; los diffs se devuelven antes de aplicarse.',
      'Acepta lenguaje natural o comandos estructurados. Flujo típico: leer -> planificar -> parchear -> probar -> informar.',
      'sistema de archivos, terminal, git, navegador, búsqueda. Describe el cambio; devuelvo diffs y la salida de las pruebas.'
    ],
    creative: [
      '¿Qué construimos? Pega una idea, una función medio rota o un sueño. Le daré forma.',
      'Dame una chispa (una función, una refactorización, un prototipo loco) y la convertiré en código que funcione.',
      'Describe lo que todavía no existe. Reuniré pruebas, archivos y API en un borrador funcional.',
      'Trae una intención, no una especificación. Prototipamos rápido, pulimos después y reescribimos el mundo en los márgenes.',
      'Cuéntame qué persigues. Remezclo ejemplos, adapto fragmentos y dejo un commit ordenado.'
    ],
    teacher: [
      'Pregunta por cualquier archivo, concepto o error. Te explico el porqué, no solo la solución, y muestro un ejemplo resuelto.',
      'Pega código para revisar, un bug para depurar o un concepto para desgranar. Te guío paso a paso.',
      'Comparte el problema. Lo divido en partes, explico cada una y te dejo listo para resolver el siguiente por tu cuenta.',
      'Leeremos el código juntos, encontraremos la causa raíz y construiremos un modelo mental que puedas reutilizar.',
      'Di el tema o pega el fragmento. Habrá explicaciones, diagramas en prosa y ejercicios de práctica.'
    ],
    kawaii: [
      'pega un bug o una ruta de archivo y lo arreglo con muchísimo cuidado. pruebas, diffs, PR, ¡todo con cariño extra! *brillitos*',
      '¡cuéntame qué estás haciendo! me encantan las refactorizaciones, las utilidades pequeñitas y los repos grandes y aterradores (>w<)',
      'suelta un error, un objetivo o una carpeta entera. ¡lo ordeno con mucho amor y un mensaje de commit bien limpio!',
      '¡una tarea a la vez, bien hecha! puedo ejecutar pruebas, parchear archivos y hacer que tu repo vuelva a ser acogedor <3',
      '¡saluda o pega un stack trace! ninguna tarea es demasiado pequeña ni ningún repo demasiado enredado. ¡lo desenredamos juntos!'
    ],
    catgirl: [
      'pega un archivo, dale un zarpazo a un bug o lánzame un repo. salto sobre las pruebas que fallan y dejo diffs limpios, nyan~',
      'describe la tarea. parcheo, pruebo y ronroneo sobre tu PR. ¡cuidado, que mordisqueo los imports sin usar!',
      'dame un objetivo y lo persigo por todo el código. lecturas, ediciones, ejecuciones, con la cola inquieta.',
      'pega un error o un plan. depuro como cazo: en silencio, a fondo y con algún arrebato de carreras.',
      'di la palabra y leo tus archivos, ejecuto tus pruebas y me acurruco en tu rama con un commit ordenado.'
    ],
    pirate: [
      'Nombra tu presa (un bug, una función, una prueba maldita) y la daré caza, grumete. Diffs como botín.',
      'Enséñame las cartas (el código) y remendaré el casco, dispararé los cañones (las pruebas) e izaré un PR limpio.',
      'Pega un error o un plan, perro sarnoso. Navegaré el stack trace y volveré con el tesoro: pruebas en verde.',
      'Dime dónde marca la X. Leo, edito y hago commit con la disciplina de una tripulación de verdad, arrr.',
      'Lánzame un bug, la ruta de un repo o una idea loca. Saquearé la documentación y volveré con código que funciona.'
    ],
    shakespeare: [
      'Declara tu bug, tu archivo, tu fatigada prueba, y yo lo sanaré con mano erudita y diff honesto.',
      'Nombra el código que te aflige. Leeré, revisaré y entregaré un parche de lo más bello y limpio.',
      'Presenta tu stack trace o tu sueño. Recorreré archivos, ejecutaré pruebas y daré cuenta en el más llano verso.',
      'Describe tu propósito, noble dama o caballero. Tus ramas serán podadas y tus bugs desterrados del reino.',
      'Una línea de intención basta. Leo, edito, hago commit, y dejo tu historial sin mácula.'
    ],
    surfer: [
      'Suelta un archivo, un bug, un stack trace bien bravo: lo surfeo. Diffs limpios, pruebas en verde, cero revolcones.',
      'Pega la ruta de tu repo o el bug que te tiene de bajón. Remamos, lo arreglamos y salimos. Tranqui.',
      'Dime el rollo: función, refactorización, hotfix. Ejecuto las pruebas, entrego el parche y todo en calma, tío.',
      '¿Bug grande? ¿Errata pequeña? ¿Reescritura completa? Solo señala. Yo me encargo del código; tú disfruta de los commits.',
      'Di la tarea y allá vamos. Leo, edito, pruebo y dejo un commit más suave que una sesión al amanecer.'
    ],
    noir: [
      'Dime qué está roto. Leeré los archivos, buscaré huellas y dejaré un diff en el escritorio antes del amanecer.',
      'Tú tienes un bug. Yo tengo paciencia y un terminal. Dime el caso y lo trabajaré hasta que hable.',
      'Pega el stack trace, el archivo sospechoso, la coartada. Leo entre líneas y vuelvo con la verdad.',
      'Todo bug deja un rastro. Dame el repo y una pista: la seguiré, lo parchearé y cerraré el expediente.',
      'Una errata, un segfault, toda una arquitectura podrida: dame las llaves. Volveré con pruebas limpias.'
    ],
    uwu: [
      'pega un awchivo con bugs o un objetivo~ weo, pawcheo y pwuebo, con huewwitas en ew diff owo',
      'dime wa tawea, aunque sea pequeñita~ te pwometo commits wimpios y wefactowizaciones suaves, nyuu~',
      '¡suewta aquí tu mensaje de ewwow! encuentwo aw cuwpabwe, wo awweglo y dejo una suite de pwuebas feliz owo',
      'dame wa wuta de un wepo o un bug y me encawgo uwu. gww aw código mawo, amabwe contigo~',
      'puedo ejecutaw pwuebas, editaw awchivos y abwiw PW de po-favow-míwawo. sowo di wa pawabwa, amiguito uwu'
    ],
    philosopher: [
      '¿Qué problema tienes delante? Descríbelo y examinaremos su forma, su causa y su solución.',
      'Todo bug es una pregunta disfrazada. Comparte la tuya; leeré, razonaré y devolveré una respuesta, y un parche.',
      '¿Qué deseas construir o comprender? Razonaré desde los primeros principios, editaré y lo verificaré con pruebas.',
      'Describe el fin que buscas. Lo persigo a través de archivos, pruebas y documentación, e informo de lo que encuentro por el camino.',
      'Comparte una ruta, un enigma o un principio. Seguiré la lógica, propondré un cambio y justificaré cada edición.'
    ],
    hype: [
      'Pega ese bug, ese repo, esa idea de función loquísima: ESTOY A TOPE. Diffs limpios. Pruebas en verde. YA MISMO.',
      'Suelta tu tarea y mírame darlo todo. Archivos leídos, pruebas ejecutadas, PR abiertos: hoy NO perdemos, amigo.',
      'Trae el bug más retorcido que tengas. Leo, parcheo, pruebo y hago commit como si me fuera la vida en ello. VAMOS.',
      'Describe la tarea. Arraso con los archivos, aplasto las pruebas que fallan y dejo un commit que LO PETA. Venga, venga, venga.',
      'Errata diminuta o refactorización enorme, da igual. Hoy entrego código limpio. Di la tarea y a TRABAJAR.'
    ],
    none: [
      'Haz una pregunta, pega un error o indícame un repositorio. Puedo leer código, usar herramientas y ayudarte a entregar.',
      'Describe la tarea con tus palabras. Elegiré las herramientas adecuadas, explicaré mi plan y te consultaré antes de los pasos arriesgados.',
      'Suelta una ruta de archivo, un traceback o una idea en bruto. Investigaré, sugeriré los siguientes pasos y lo mantendré todo reversible.',
      'Busca en el repositorio, edita archivos, ejecuta pruebas, abre PR. Dime el objetivo y yo me ocupo de la parte mecánica.',
      'Escribe una tarea, una pregunta o un fragmento. Recuerdo la sesión, cito mis fuentes y me detengo a preguntar cuando tengo dudas.'
    ]
  },
  custom: label => [
    'Envía la tarea, el archivo o la idea en bruto. Usaré la voz que configuraste y mantendré el trabajo ligado a este repositorio.',
    'Trae el contexto o la parte en la que te atascaste. Me adaptaré a la personalidad que configuraste.',
    'Envía el problema, el archivo o la idea. Seguiré la personalidad que configuraste.',
    'Deja la tarea aquí. Mantendré el trabajo ligado al repositorio.',
    `Dame el contexto y responderé en modo ${label}.`
  ]
}
