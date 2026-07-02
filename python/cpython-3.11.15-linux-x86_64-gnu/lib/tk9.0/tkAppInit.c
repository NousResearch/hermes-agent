/*
 * tkAppInit.c --
 *
 *	Provides a default version of the main program and Tcl_AppInit
 *	procedure for wish and other Tk-based applications.
 *
 * Copyright © 1993 The Regents of the University of California.
 * Copyright © 1994-1997 Sun Microsystems, Inc.
 * Copyright © 1998-1999 Scriptics Corporation.
 *
 * See the file "license.terms" for information on usage and redistribution of
 * this file, and for a DISCLAIMER OF ALL WARRANTIES.
 */

/*
 * Explanation on following undef USE_TCL_STUBS by JN 2023-12-19 on the core list:
 * What's going on is related to TIP #596:
 *  Stubs support for Embedding Tcl in other applications
 *
 * If an application using Tcl_Main() is compiled with USE_TCL_STUBS,
 * Tcl_Main() will be replaced by a stub function, which loads
 * libtcl9.0.so/tcl90.dll and then calls its Tcl_MainEx(). If
 * libtcl9.0.so/tcl90.dll is not present (at runtime), a crash is what happens.
 *
 * So ... tkAppInit.c should not be compiled with USE_TCL_STUBS
 * (unless you want to use the TIP #596 functionality)
 *
 * The proper solution is to make sure that Makefile.in doesn't use
 * TCL_USE_STUBS when compiling tkAppInit.c. But that's a
 * quite big re-organization just before a b1 release. Simpler
 * is just to #undef'ine USE_TCL_STUBS, it has the same effect.
 */
#undef USE_TCL_STUBS
#undef BUILD_tk
#undef STATIC_BUILD
#include "tk.h"
#include "tkPort.h"
#if (TCL_MAJOR_VERSION < 9)
#   define Tcl_LibraryInitProc Tcl_PackageInitProc
#   define Tcl_StaticLibrary Tcl_StaticPackage
#endif

#ifdef TK_TEST
#ifdef __cplusplus
extern "C" {
#endif
extern Tcl_LibraryInitProc Tktest_Init;
#ifdef __cplusplus
}
#endif
#endif /* TK_TEST */

/*
 * The following #if block allows you to change the AppInit function by using
 * a #define of TCL_LOCAL_APPINIT instead of rewriting this entire file. The
 * #if checks for that #define and uses Tcl_AppInit if it doesn't exist.
 */

#ifndef TK_LOCAL_APPINIT
#define TK_LOCAL_APPINIT Tcl_AppInit
#endif
#ifndef MODULE_SCOPE
#   ifdef __cplusplus
#	define MODULE_SCOPE extern "C"
#   else
#	define MODULE_SCOPE extern
#   endif
#endif
MODULE_SCOPE int TK_LOCAL_APPINIT(Tcl_Interp *);
MODULE_SCOPE int main(int, char **);

/*
 * The following #if block allows you to change how Tcl finds the startup
 * script, prime the library or encoding paths, fiddle with the argv, etc.,
 * without needing to rewrite Tk_Main()
 */

#ifdef TK_LOCAL_MAIN_HOOK
MODULE_SCOPE int TK_LOCAL_MAIN_HOOK(int *argc, char ***argv);
#endif

/* Make sure the stubbed variants of those are never used. */
#undef Tcl_ObjSetVar2
#undef Tcl_NewStringObj

/*
 *----------------------------------------------------------------------
 *
 * main --
 *
 *	This is the main program for the application.
 *
 * Results:
 *	None: Tk_Main never returns here, so this procedure never returns
 *	either.
 *
 * Side effects:
 *	Just about anything, since from here we call arbitrary Tcl code.
 *
 *----------------------------------------------------------------------
 */

int
main(
    int argc,			/* Number of command-line arguments. */
    char **argv)		/* Values of command-line arguments. */
{
#ifdef TK_LOCAL_MAIN_HOOK
    TK_LOCAL_MAIN_HOOK(&argc, &argv);
#elif TCL_MAJOR_VERSION > 8
    /* This doesn't work with Tcl 8.6 */
    TclZipfs_AppHook(&argc, &argv);
#endif

    Tk_Main(argc, argv, TK_LOCAL_APPINIT);
    return 0;			/* Needed only to prevent compiler warning. */
}

/*
 *----------------------------------------------------------------------
 *
 * Tcl_AppInit --
 *
 *	This procedure performs application-specific initialization. Most
 *	applications, especially those that incorporate additional packages,
 *	will have their own version of this procedure.
 *
 * Results:
 *	Returns a standard Tcl completion code, and leaves an error message in
 *	the interp's result if an error occurs.
 *
 * Side effects:
 *	Depends on the startup script.
 *
 *----------------------------------------------------------------------
 */

int
Tcl_AppInit(
    Tcl_Interp *interp)		/* Interpreter for application. */
{
    if ((Tcl_Init)(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }

    if (Tk_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    Tcl_StaticLibrary(interp, "Tk", Tk_Init, Tk_SafeInit);

#if defined(USE_CUSTOM_EXIT_PROC)
    if (TkpWantsExitProc()) {
	Tcl_SetExitProc(TkpExitProc);
    }
#endif

#ifdef TK_TEST
    if (Tktest_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    Tcl_StaticLibrary(interp, "Tktest", Tktest_Init, 0);
#endif /* TK_TEST */

    /*
     * Call the init procedures for included packages. Each call should look
     * like this:
     *
     * if (Mod_Init(interp) == TCL_ERROR) {
     *     return TCL_ERROR;
     * }
     *
     * where "Mod" is the name of the module. (Dynamically-loadable packages
     * should have the same entry-point name.)
     */

    /*
     * Call Tcl_CreateObjCommand for application-specific commands, if they
     * weren't already created by the init procedures called above.
     */

    /*
     * Specify a user-specific startup file to invoke if the application is
     * run interactively. Typically the startup file is "~/.apprc" where "app"
     * is the name of the application. If this line is deleted then no
     * user-specific startup file will be run under any conditions.
     */

    (void) Tcl_EvalEx(interp,
	    "set tcl_rcFileName [file tildeexpand ~/.wishrc]",
	    -1, TCL_EVAL_GLOBAL);
    return TCL_OK;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
