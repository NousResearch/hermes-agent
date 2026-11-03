/** Copy for group chats whose members live on more than one connected machine.
 *  Such a room runs on this Desktop (each member answers from its own gateway),
 *  so the dialog says so before it is created. Every Bot Mode locale. */
export interface CrossGatewayGroupMessages {
  desktopDriven: string
  /** A bot's machine is offline: Manage groups cannot change its rooms. */
  groupsOffline: (connection: string, bot: string) => string
  /** A room member's machine is offline: Manage members cannot remove it. */
  memberOffline: (connection: string, bot: string) => string
  /** Save stopped because a member's machine did not take the change. */
  memberSaveFailed: (connection: string, bot: string) => string
  /** Save stopped, and putting back a change already made on a member's machine did not land. */
  memberRollbackFailed: (connection: string, bot: string) => string
  selectionElsewhere: (connection: string) => string
}

export const CROSS_GATEWAY_GROUP_LOCALES = {
  en: {
    desktopDriven:
      'Runs while this Desktop is open: bots on other machines answer from their own gateways, and this Desktop carries the messages between them.',
    groupsOffline: (connection, bot) => `${connection} is offline — reconnect to change ${bot}’s groups.`,
    memberOffline: (connection, bot) => `${connection} is offline — reconnect to remove ${bot}.`,
    memberSaveFailed: (connection, bot) => `Members not saved: ${connection} did not take the change for ${bot}. Nothing was changed; try again once it is reachable.`,
    memberRollbackFailed: (connection, bot) =>
      `Members not saved, and the change already made for ${bot} on ${connection} could not be undone: ${connection} may still have it. The room shows what ${connection} last reported; reconnect it and check the members.`,
    selectionElsewhere: connection => `These bots all live on ${connection}. Add a bot from this connection, or switch to ${connection} to start a group of its bots.`
  },
  ja: {
    desktopDriven:
      'このデスクトップを開いている間だけ動作します。ほかのマシンのボットはそれぞれのゲートウェイから返信し、このデスクトップがメッセージを中継します。',
    groupsOffline: (connection, bot) => `${connection} はオフラインです。${bot} のグループを変更するには再接続してください。`,
    memberOffline: (connection, bot) => `${connection} はオフラインです。${bot} を外すには再接続してください。`,
    memberSaveFailed: (connection, bot) => `メンバーを保存できませんでした: ${connection} が ${bot} の変更を受け付けませんでした。何も変更されていません。接続できるようになってから再試行してください。`,
    memberRollbackFailed: (connection, bot) =>
      `メンバーを保存できませんでした。${connection} 上の ${bot} にすでに加えた変更を元に戻せませんでした。${connection} には変更が残っている可能性があります。ルームには ${connection} が最後に報告した状態を表示しています。再接続してメンバーを確認してください。`,
    selectionElsewhere: connection => `これらのボットはすべて ${connection} にあります。この接続のボットを追加するか、${connection} に切り替えてそのボットだけのグループを作成してください。`
  },
  zh: {
    desktopDriven: '仅在此桌面端打开时运行：其他机器上的机器人通过各自的网关回复，由此桌面端在它们之间传递消息。',
    groupsOffline: (connection, bot) => `${connection} 已离线——重新连接后才能更改 ${bot} 的群组。`,
    memberOffline: (connection, bot) => `${connection} 已离线——重新连接后才能移除 ${bot}。`,
    memberSaveFailed: (connection, bot) => `成员未保存：${connection} 未接受 ${bot} 的更改。未做任何更改；请在可连接后重试。`,
    memberRollbackFailed: (connection, bot) =>
      `成员未保存，且已对 ${connection} 上的 ${bot} 做出的更改无法撤销：${connection} 可能仍保留该更改。房间显示的是 ${connection} 最后报告的状态；请重新连接并检查成员。`,
    selectionElsewhere: connection => `这些机器人都在 ${connection} 上。请添加一个此连接上的机器人，或切换到 ${connection} 来创建只包含其机器人的群组。`
  },
  'zh-hant': {
    desktopDriven: '僅在此桌面端開啟時執行：其他機器上的機器人透過各自的閘道回覆，由此桌面端在它們之間傳遞訊息。',
    groupsOffline: (connection, bot) => `${connection} 已離線——重新連線後才能變更 ${bot} 的群組。`,
    memberOffline: (connection, bot) => `${connection} 已離線——重新連線後才能移除 ${bot}。`,
    memberSaveFailed: (connection, bot) => `成員未儲存：${connection} 未接受 ${bot} 的變更。未做任何變更；請在可連線後重試。`,
    memberRollbackFailed: (connection, bot) =>
      `成員未儲存，且已對 ${connection} 上的 ${bot} 做出的變更無法復原：${connection} 可能仍保留該變更。房間顯示的是 ${connection} 最後回報的狀態；請重新連線並檢查成員。`,
    selectionElsewhere: connection => `這些機器人都在 ${connection} 上。請加入一個此連線上的機器人，或切換到 ${connection} 來建立只包含其機器人的群組。`
  },
  ar: {
    desktopDriven:
      'تعمل ما دام سطح المكتب هذا مفتوحًا: تردّ الروبوتات على الأجهزة الأخرى من بواباتها الخاصة، وينقل سطح المكتب هذا الرسائل بينها.',
    groupsOffline: (connection, bot) => `${connection} غير متصل — أعد الاتصال لتغيير مجموعات ${bot}.`,
    memberOffline: (connection, bot) => `${connection} غير متصل — أعد الاتصال لإزالة ${bot}.`,
    memberSaveFailed: (connection, bot) => `لم تُحفظ الأعضاء: لم يقبل ${connection} التغيير الخاص بـ ${bot}. لم يتغيّر شيء؛ أعد المحاولة عندما يصبح متاحًا.`,
    memberRollbackFailed: (connection, bot) =>
      `لم تُحفظ الأعضاء، وتعذّر التراجع عن التغيير الذي أُجري بالفعل لـ ${bot} على ${connection}: قد يظل التغيير موجودًا على ${connection}. تعرض الغرفة آخر ما أبلغ عنه ${connection}؛ أعد الاتصال به وتحقق من الأعضاء.`,
    selectionElsewhere: connection => `كل هذه الروبوتات موجودة على ${connection}. أضف روبوتًا من هذا الاتصال، أو انتقل إلى ${connection} لبدء مجموعة من روبوتاته.`
  },
  ru: {
    desktopDriven:
      'Работает, пока открыт этот Desktop: боты на других машинах отвечают через свои шлюзы, а этот Desktop передаёт сообщения между ними.',
    groupsOffline: (connection, bot) => `${connection} не в сети — переподключитесь, чтобы изменить группы ${bot}.`,
    memberOffline: (connection, bot) => `${connection} не в сети — переподключитесь, чтобы убрать ${bot}.`,
    memberSaveFailed: (connection, bot) => `Участники не сохранены: ${connection} не принял изменение для ${bot}. Ничего не изменено; повторите, когда он станет доступен.`,
    memberRollbackFailed: (connection, bot) =>
      `Участники не сохранены, и уже внесённое изменение для ${bot} на ${connection} не удалось отменить: на ${connection} оно может остаться. Комната показывает последнее, что сообщил ${connection}; переподключитесь и проверьте участников.`,
    selectionElsewhere: connection => `Все эти боты находятся на ${connection}. Добавьте бота из этого подключения или переключитесь на ${connection}, чтобы создать группу из его ботов.`
  },
  fr: {
    desktopDriven:
      'Fonctionne tant que ce Desktop est ouvert : les bots des autres machines répondent depuis leur propre passerelle, et ce Desktop transmet les messages entre eux.',
    groupsOffline: (connection, bot) => `${connection} est hors ligne — reconnectez-le pour modifier les groupes de ${bot}.`,
    memberOffline: (connection, bot) => `${connection} est hors ligne — reconnectez-le pour retirer ${bot}.`,
    memberSaveFailed: (connection, bot) => `Membres non enregistrés : ${connection} n’a pas pris la modification pour ${bot}. Rien n’a changé ; réessayez quand il sera joignable.`,
    memberRollbackFailed: (connection, bot) =>
      `Membres non enregistrés, et la modification déjà faite pour ${bot} sur ${connection} n’a pas pu être annulée : ${connection} l’a peut-être encore. Le salon affiche le dernier état signalé par ${connection} ; reconnectez-le et vérifiez les membres.`,
    selectionElsewhere: connection => `Ces bots se trouvent tous sur ${connection}. Ajoutez un bot de cette connexion, ou passez sur ${connection} pour créer un groupe avec ses bots.`
  },
  de: {
    desktopDriven:
      'Läuft, solange dieser Desktop geöffnet ist: Bots auf anderen Rechnern antworten über ihr eigenes Gateway, und dieser Desktop leitet die Nachrichten zwischen ihnen weiter.',
    groupsOffline: (connection, bot) => `${connection} ist offline – verbinde es neu, um die Gruppen von ${bot} zu ändern.`,
    memberOffline: (connection, bot) => `${connection} ist offline – verbinde es neu, um ${bot} zu entfernen.`,
    memberSaveFailed: (connection, bot) => `Mitglieder nicht gespeichert: ${connection} hat die Änderung für ${bot} nicht übernommen. Nichts wurde geändert; versuche es erneut, sobald es erreichbar ist.`,
    memberRollbackFailed: (connection, bot) =>
      `Mitglieder nicht gespeichert, und die bereits vorgenommene Änderung für ${bot} auf ${connection} ließ sich nicht rückgängig machen: ${connection} hat sie möglicherweise noch. Der Raum zeigt den zuletzt von ${connection} gemeldeten Stand; verbinde es neu und prüfe die Mitglieder.`,
    selectionElsewhere: connection => `Diese Bots liegen alle auf ${connection}. Füge einen Bot dieser Verbindung hinzu oder wechsle zu ${connection}, um eine Gruppe mit dessen Bots zu starten.`
  },
  es: {
    desktopDriven:
      'Funciona mientras este Desktop esté abierto: los bots de otras máquinas responden desde su propia pasarela y este Desktop transmite los mensajes entre ellos.',
    groupsOffline: (connection, bot) => `${connection} está sin conexión: vuelve a conectarlo para cambiar los grupos de ${bot}.`,
    memberOffline: (connection, bot) => `${connection} está sin conexión: vuelve a conectarlo para quitar a ${bot}.`,
    memberSaveFailed: (connection, bot) => `Miembros no guardados: ${connection} no aceptó el cambio de ${bot}. No se cambió nada; inténtalo de nuevo cuando esté disponible.`,
    memberRollbackFailed: (connection, bot) =>
      `Miembros no guardados, y el cambio ya hecho para ${bot} en ${connection} no se pudo deshacer: puede que ${connection} todavía lo tenga. La sala muestra lo último que informó ${connection}; vuelve a conectarlo y revisa los miembros.`,
    selectionElsewhere: connection => `Todos estos bots están en ${connection}. Añade un bot de esta conexión o cambia a ${connection} para crear un grupo con sus bots.`
  }
} satisfies Record<string, CrossGatewayGroupMessages>
