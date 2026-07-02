#ifndef Py_INTERNAL_TRACEBACK_H
#define Py_INTERNAL_TRACEBACK_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/* Write the Python traceback into the file 'fd'. For example:

       Traceback (most recent call first):
         File "xxx", line xxx in <xxx>
         File "xxx", line xxx in <xxx>
         ...
         File "xxx", line xxx in <xxx>

   This function is written for debug purpose only, to dump the traceback in
   the worst case: after a segmentation fault, at fatal error, etc. That's why,
   it is very limited. Strings are truncated to 100 characters and encoded to
   ASCII with backslashreplace. It doesn't write the source code, only the
   function name, filename and line number of each frame. Write only the first
   100 frames: if the traceback is truncated, write the line " ...".

   This function is signal safe. */

PyAPI_FUNC(void) _Py_DumpTraceback(
    int fd,
    PyThreadState *tstate);

/* Write the traceback of all threads into the file 'fd'. current_thread can be
   NULL.

   Return NULL on success, or an error message on error.

   This function is written for debug purpose only. It calls
   _Py_DumpTraceback() for each thread, and so has the same limitations. It
   only write the traceback of the first 100 threads: write "..." if there are
   more threads.

   If current_tstate is NULL, the function tries to get the Python thread state
   of the current thread. It is not an error if the function is unable to get
   the current Python thread state.

   If interp is NULL, the function tries to get the interpreter state from
   the current Python thread state, or from
   _PyGILState_GetInterpreterStateUnsafe() in last resort.

   It is better to pass NULL to interp and current_tstate, the function tries
   different options to retrieve this information.

   This function is signal safe. */

PyAPI_FUNC(const char*) _Py_DumpTracebackThreads(
    int fd,
    PyInterpreterState *interp,
    PyThreadState *current_tstate);

/* Write a Unicode object into the file descriptor fd. Encode the string to
   ASCII using the backslashreplace error handler.

   Do nothing if text is not a Unicode object. The function accepts Unicode
   string which is not ready (PyUnicode_WCHAR_KIND).

   This function is signal safe. */
PyAPI_FUNC(void) _Py_DumpASCII(int fd, PyObject *text);

/* Format an integer as decimal into the file descriptor fd.

   This function is signal safe. */
PyAPI_FUNC(void) _Py_DumpDecimal(
    int fd,
    size_t value);

/* Format an integer as hexadecimal with width digits into fd file descriptor.
   The function is signal safe. */
PyAPI_FUNC(void) _Py_DumpHexadecimal(
    int fd,
    uintptr_t value,
    Py_ssize_t width);

PyAPI_FUNC(PyObject*) _PyTraceBack_FromFrame(
    PyObject *tb_next,
    PyFrameObject *frame);

#define EXCEPTION_TB_HEADER "Traceback (most recent call last):\n"
#define EXCEPTION_GROUP_TB_HEADER "Exception Group Traceback (most recent call last):\n"

/* Write the traceback tb to file f. Prefix each line with
   indent spaces followed by the margin (if it is not NULL). */
PyAPI_FUNC(int) _PyTraceBack_Print_Indented(
    PyObject *tb, int indent, const char* margin,
    const char *header_margin, const char *header, PyObject *f);
PyAPI_FUNC(int) _Py_WriteIndentedMargin(int, const char*, PyObject *);
PyAPI_FUNC(int) _Py_WriteIndent(int, PyObject *);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_TRACEBACK_H */
