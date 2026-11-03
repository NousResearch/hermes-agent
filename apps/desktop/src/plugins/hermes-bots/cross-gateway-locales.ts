/** Copy for group chats whose members live on more than one connected machine.
 *  Such a room runs on this Desktop (each member answers from its own gateway),
 *  so the dialog says so before it is created. Every Bot Mode locale. */
export interface CrossGatewayGroupMessages {
  desktopDriven: string
  selectionElsewhere: (connection: string) => string
}

export const CROSS_GATEWAY_GROUP_LOCALES = {
  en: {
    desktopDriven:
      'Runs while this Desktop is open: bots on other machines answer from their own gateways, and this Desktop carries the messages between them.',
    selectionElsewhere: connection => `These bots all live on ${connection}. Add a bot from this connection, or switch to ${connection} to start a group of its bots.`
  },
  ja: {
    desktopDriven:
      'このデスクトップを開いている間だけ動作します。ほかのマシンのボットはそれぞれのゲートウェイから返信し、このデスクトップがメッセージを中継します。',
    selectionElsewhere: connection => `これらのボットはすべて ${connection} にあります。この接続のボットを追加するか、${connection} に切り替えてそのボットだけのグループを作成してください。`
  },
  zh: {
    desktopDriven: '仅在此桌面端打开时运行：其他机器上的机器人通过各自的网关回复，由此桌面端在它们之间传递消息。',
    selectionElsewhere: connection => `这些机器人都在 ${connection} 上。请添加一个此连接上的机器人，或切换到 ${connection} 来创建只包含其机器人的群组。`
  },
  'zh-hant': {
    desktopDriven: '僅在此桌面端開啟時執行：其他機器上的機器人透過各自的閘道回覆，由此桌面端在它們之間傳遞訊息。',
    selectionElsewhere: connection => `這些機器人都在 ${connection} 上。請加入一個此連線上的機器人，或切換到 ${connection} 來建立只包含其機器人的群組。`
  },
  ar: {
    desktopDriven:
      'تعمل ما دام سطح المكتب هذا مفتوحًا: تردّ الروبوتات على الأجهزة الأخرى من بواباتها الخاصة، وينقل سطح المكتب هذا الرسائل بينها.',
    selectionElsewhere: connection => `كل هذه الروبوتات موجودة على ${connection}. أضف روبوتًا من هذا الاتصال، أو انتقل إلى ${connection} لبدء مجموعة من روبوتاته.`
  },
  ru: {
    desktopDriven:
      'Работает, пока открыт этот Desktop: боты на других машинах отвечают через свои шлюзы, а этот Desktop передаёт сообщения между ними.',
    selectionElsewhere: connection => `Все эти боты находятся на ${connection}. Добавьте бота из этого подключения или переключитесь на ${connection}, чтобы создать группу из его ботов.`
  },
  fr: {
    desktopDriven:
      'Fonctionne tant que ce Desktop est ouvert : les bots des autres machines répondent depuis leur propre passerelle, et ce Desktop transmet les messages entre eux.',
    selectionElsewhere: connection => `Ces bots se trouvent tous sur ${connection}. Ajoutez un bot de cette connexion, ou passez sur ${connection} pour créer un groupe avec ses bots.`
  },
  de: {
    desktopDriven:
      'Läuft, solange dieser Desktop geöffnet ist: Bots auf anderen Rechnern antworten über ihr eigenes Gateway, und dieser Desktop leitet die Nachrichten zwischen ihnen weiter.',
    selectionElsewhere: connection => `Diese Bots liegen alle auf ${connection}. Füge einen Bot dieser Verbindung hinzu oder wechsle zu ${connection}, um eine Gruppe mit dessen Bots zu starten.`
  },
  es: {
    desktopDriven:
      'Funciona mientras este Desktop esté abierto: los bots de otras máquinas responden desde su propia pasarela y este Desktop transmite los mensajes entre ellos.',
    selectionElsewhere: connection => `Todos estos bots están en ${connection}. Añade un bot de esta conexión o cambia a ${connection} para crear un grupo con sus bots.`
  }
} satisfies Record<string, CrossGatewayGroupMessages>
