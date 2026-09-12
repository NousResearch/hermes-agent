import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import 'design/hermes_theme.dart';
import 'features/chat/presentation/chat_screen.dart';
import 'features/connections/presentation/add_connection_screen.dart';
import 'features/instances/presentation/instance_list_screen.dart';
import 'features/sessions/presentation/sessions_screen.dart';

final _router = GoRouter(
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const InstanceListScreen(),
      routes: [
        GoRoute(
          path: 'connections/new',
          builder: (context, state) => const AddConnectionScreen(),
        ),
        GoRoute(
          path: 'sessions',
          builder: (context, state) => const SessionsScreen(),
        ),
        GoRoute(
          path: 'chat',
          builder: (context, state) => const ChatScreen(),
        ),
      ],
    ),
  ],
);

class HermesRemoteClientApp extends StatelessWidget {
  const HermesRemoteClientApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'Hermes Remote',
      theme: hermesTheme(Brightness.light),
      darkTheme: hermesTheme(Brightness.dark),
      themeMode: ThemeMode.system,
      routerConfig: _router,
    );
  }
}
