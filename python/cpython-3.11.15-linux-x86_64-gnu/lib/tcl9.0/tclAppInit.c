/*
 * tclAppInit.c --
 *
 *	Provides a default version of the main program and Tcl_AppInit
 *	procedure for tclsh and other Tcl-based applications (without Tk).
 *
 * Copyright (c) 1993 The Regents of the University of California.
 * Copyright (c) 1994-1997 Sun Microsystems, Inc.
 * Copyright (c) 1998-1999 Scriptics Corporation.
 *
 * See the file "license.terms" for information on usage and redistribution of
 * this file, and for a DISCLAIMER OF ALL WARRANTIES.
 */

#include "tcl.h"
#if TCL_MAJOR_VERSION < 9
#   if defined(USE_TCL_STUBS)
#	error "Don't build with USE_TCL_STUBS!"
#   endif
#   define Tcl_LibraryInitProc Tcl_PackageInitProc
#   define Tcl_StaticLibrary Tcl_StaticPackage
#endif

#ifdef TCL_TEST
#ifdef __cplusplus
extern "C" {
#endif
extern Tcl_LibraryInitProc Tcltest_Init;
extern Tcl_LibraryInitProc Tcltest_SafeInit;
#ifdef __cplusplus
}
#endif
#endif /* TCL_TEST */

#ifdef TCL_XT_TEST
extern void                XtToolkitInitialize(void);
extern Tcl_LibraryInitProc Tclxttest_Init;
#endif /* TCL_XT_TEST */

/*
 * The following #if block allows you to change the AppInit function by using
 * a #define of TCL_LOCAL_APPINIT instead of rewriting this entire file. The
 * #if checks for that #define and uses Tcl_AppInit if it does not exist.
 */

#ifndef TCL_LOCAL_APPINIT
#define TCL_LOCAL_APPINIT Tcl_AppInit
#endif
#ifndef MODULE_SCOPE
#   define MODULE_SCOPE extern
#endif
MODULE_SCOPE int TCL_LOCAL_APPINIT(Tcl_Interp *);
MODULE_SCOPE int main(int, char **);

/*
 * The following #if block allows you to change how Tcl finds the startup
 * script, prime the library or encoding paths, fiddle with the argv, etc.,
 * without needing to rewrite Tcl_Main()
 */

#ifdef TCL_LOCAL_MAIN_HOOK
MODULE_SCOPE int TCL_LOCAL_MAIN_HOOK(int *argc, char ***argv);
#endif

/*
 *----------------------------------------------------------------------
 *
 * main --
 *
 *	This is the main program for the application.
 *
 * Results:
 *	None: Tcl_Main never returns here, so this procedure never returns
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
    char *argv[])		/* Values of command-line arguments. */
{
#ifdef TCL_XT_TEST
    XtToolkitInitialize();
#endif

#ifdef TCL_LOCAL_MAIN_HOOK
    TCL_LOCAL_MAIN_HOOK(&argc, &argv);
#elif TCL_MAJOR_VERSION > 8 && (!defined(_WIN32) || defined(UNICODE))
    /* New in Tcl 9.0. This doesn't work on Windows without UNICODE */
    TclZipfs_AppHook(&argc, &argv);
#endif

    Tcl_Main(argc, argv, TCL_LOCAL_APPINIT);
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
    if (Tcl_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }

#ifdef TCL_XT_TEST
    if (Tclxttest_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
#endif

#ifdef TCL_TEST
    if (Tcltest_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    Tcl_StaticLibrary(interp, "Tcltest", Tcltest_Init, Tcltest_SafeInit);
#endif /* TCL_TEST */

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
#ifdef DJGPP
#define INITFILENAME "tclshrc.tcl"
#else
#define INITFILENAME ".tclshrc"
#endif

    (void) Tcl_EvalEx(interp,
	    "set tcl_rcFileName [file tildeexpand ~/" INITFILENAME "]",
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
